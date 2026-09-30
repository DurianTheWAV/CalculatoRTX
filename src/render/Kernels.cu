// CalculatoRTX - kernels CUDA : environnement HDR procédural, conversions, post-traitement.
#include "Kernels.h"

#include "../common/Check.h"
#include "../common/VecMath.h"

namespace crtx {

namespace {

// ============================================================ environnement "studio"
__device__ float3 studio(float3 d)
{
    const float y = d.y;
    const float3 sky = lerp(make_float3(0.010f, 0.011f, 0.014f), make_float3(0.030f, 0.034f, 0.044f),
                            smoothstepf(0.0f, 0.8f, y));
    const float3 ground = make_float3(0.006f, 0.0055f, 0.005f);
    float3 c = y > 0.0f ? sky : ground;
    c += make_float3(0.018f, 0.020f, 0.026f) * expf(-fabsf(y) * 9.0f);  // halo d'horizon
    // grand réflecteur diffus au plafond (reflets doux sur les surfaces vernies)
    const float k1 = dot(d, normalize(make_float3(-0.25f, 0.92f, 0.3f)));
    c += make_float3(0.20f, 0.195f, 0.185f) * smoothstepf(0.90f, 0.975f, k1);
    // bandeau vert à l'horizon arrière (clin d'œil "RTX")
    if (d.z < 0.0f) {
        const float t = (y - 0.12f) / 0.035f;
        c += make_float3(0.06f, 0.30f, 0.0f) * (expf(-t * t) * smoothstepf(0.2f, 0.9f, -d.z));
    }
    // fenêtre chaude latérale
    const float k2 = dot(d, normalize(make_float3(0.95f, 0.25f, 0.1f)));
    c += make_float3(0.35f, 0.24f, 0.14f) * smoothstepf(0.965f, 0.985f, k2);
    return c;
}

__global__ void bakeEnvKernel(float4* out, int w, int h)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float u = (x + 0.5f) / w, v = (y + 0.5f) / h;
    const float phi = (u - 0.5f) * kTwoPi, theta = v * kPi;
    const float3 d = make_float3(sinf(theta) * sinf(phi), cosf(theta), -sinf(theta) * cosf(phi));
    out[y * w + x] = make_float4v(studio(d), 1.0f);
}

__global__ void packHalf4Kernel(const float4* __restrict__ in, __half2* __restrict__ out, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float4 c = in[i];
    auto fix = [](float v) { return (v == v && v < 65000.0f) ? fmaxf(v, 0.0f) : 0.0f; };
    c.x = fix(c.x);
    c.y = fix(c.y);
    c.z = fix(c.z);
    out[2 * i + 0] = __floats2half2_rn(c.x, c.y);
    out[2 * i + 1] = __floats2half2_rn(c.z, 1.0f);
}

// ============================================================ post-traitement
__device__ __forceinline__ float3 loadColor(const void* p, bool half, int i)
{
    if (half) {
        const __half2* h = reinterpret_cast<const __half2*>(p);
        const float2 a = __half22float2(h[2 * i]);
        const float2 b = __half22float2(h[2 * i + 1]);
        return make_float3(a.x, a.y, b.x);
    }
    const float4 c = reinterpret_cast<const float4*>(p)[i];
    return make_float3(c.x, c.y, c.z);
}

__global__ void bloomDownsample(const void* in, bool half, int w, int h, float4* out, int bw, int bh,
                                const PostProcessor::Params* prm)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bw || y >= bh) return;
    const float thr = prm->bloomThreshold;
    const float ex = prm->exposure;
    float3 sum = make_float3(0, 0, 0);
    for (int j = 0; j < 4; ++j) {
        for (int i = 0; i < 4; ++i) {
            const int sx = min(x * 4 + i, w - 1), sy = min(y * 4 + j, h - 1);
            float3 c = loadColor(in, half, sy * w + sx) * ex;
            const float l = luminance(c);
            const float k = fmaxf(l - thr, 0.0f) / fmaxf(l, 1e-4f);
            sum += c * k;
        }
    }
    out[y * bw + x] = make_float4v(sum * (1.0f / 16.0f), 1.0f);
}

__constant__ float kGauss[13] = {0.0450f, 0.0605f, 0.0763f, 0.0909f, 0.1024f, 0.1091f, 0.1117f,
                                 0.1091f, 0.1024f, 0.0909f, 0.0763f, 0.0605f, 0.0450f};

__global__ void bloomBlur(const float4* in, float4* out, int bw, int bh, int dx, int dy)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= bw || y >= bh) return;
    float3 s = make_float3(0, 0, 0);
    float wsum = 0.0f;
    for (int k = -6; k <= 6; ++k) {
        const int sx = min(max(x + k * dx * 2, 0), bw - 1);
        const int sy = min(max(y + k * dy * 2, 0), bh - 1);
        const float wgt = kGauss[k + 6];
        s += xyz(in[sy * bw + sx]) * wgt;
        wsum += wgt;
    }
    out[y * bw + x] = make_float4v(s * (1.0f / wsum), 1.0f);
}

__device__ __forceinline__ float3 acesFilm(float3 x)
{
    // approximation de Narkowicz (ACES filmique)
    const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
    auto f = [&](float v) { return saturate((v * (a * v + b)) / (v * (c * v + d) + e)); };
    return make_float3(f(x.x), f(x.y), f(x.z));
}

__device__ __forceinline__ float linearToSrgb(float v)
{
    v = saturate(v);
    return v <= 0.0031308f ? 12.92f * v : 1.055f * powf(v, 1.0f / 2.4f) - 0.055f;
}

__device__ float3 sampleBloom(const float4* b, int bw, int bh, float u, float v)
{
    const float fx = u * bw - 0.5f, fy = v * bh - 0.5f;
    const int x0 = max(0, min(bw - 1, static_cast<int>(floorf(fx))));
    const int y0 = max(0, min(bh - 1, static_cast<int>(floorf(fy))));
    const int x1 = min(bw - 1, x0 + 1), y1 = min(bh - 1, y0 + 1);
    const float tx = saturate(fx - x0), ty = saturate(fy - y0);
    const float3 a = lerp(xyz(b[y0 * bw + x0]), xyz(b[y0 * bw + x1]), tx);
    const float3 c = lerp(xyz(b[y1 * bw + x0]), xyz(b[y1 * bw + x1]), tx);
    return lerp(a, c, ty);
}

__global__ void composite(const void* in, bool half, const float4* bloom, int bw, int bh, uchar4* out, int w, int h,
                          bool bgra, const PostProcessor::Params* prm)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const float u = (x + 0.5f) / w, v = (y + 0.5f) / h;
    float3 c = loadColor(in, half, y * w + x) * prm->exposure;
    c += sampleBloom(bloom, bw, bh, u, v) * prm->bloomStrength;  // fraction de l'énergie au-dessus du seuil
    // vignettage optique léger
    const float dx = u - 0.5f, dy = v - 0.5f;
    c = c * (1.0f - prm->vignette * (dx * dx + dy * dy) * 2.0f);
    c = acesFilm(c);
    // tramage (bruit bleu approché) pour éviter les bandes en 8 bits
    const unsigned hsh = (x * 1973u + y * 9277u + prm->frame * 26699u) | 1u;
    const unsigned r = (hsh ^ (hsh >> 13)) * 0x5bd1e995u;
    const float dither = ((r >> 8) & 0xFF) / 255.0f - 0.5f;
    const float R = linearToSrgb(c.x) * 255.0f + dither;
    const float G = linearToSrgb(c.y) * 255.0f + dither;
    const float B = linearToSrgb(c.z) * 255.0f + dither;
    auto q = [](float s) { return static_cast<unsigned char>(fminf(fmaxf(s + 0.5f, 0.0f), 255.0f)); };
    out[y * w + x] = bgra ? make_uchar4(q(B), q(G), q(R), 255) : make_uchar4(q(R), q(G), q(B), 255);
}

}  // namespace

namespace kernels {

void bakeEnvironment(float4* out, int width, int height, cudaStream_t stream)
{
    const dim3 block(16, 16);
    const dim3 grid((width + 15) / 16, (height + 15) / 16);
    bakeEnvKernel<<<grid, block, 0, stream>>>(out, width, height);
    CUDA_CHECK_LAST();
}

void packHalf4(const float4* in, __half* out, int count, cudaStream_t stream)
{
    packHalf4Kernel<<<(count + 255) / 256, 256, 0, stream>>>(in, reinterpret_cast<__half2*>(out), count);
    CUDA_CHECK_LAST();
}

}  // namespace kernels

PostProcessor::PostProcessor()
{
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParams_), sizeof(Params)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hParams_), sizeof(Params)));
}

PostProcessor::~PostProcessor()
{
    destroyGraph();
    cudaFree(bloomA_);
    cudaFree(bloomB_);
    cudaFree(dParams_);
    cudaFreeHost(hParams_);
}

void PostProcessor::destroyGraph()
{
    if (graphExec_) cudaGraphExecDestroy(graphExec_);
    if (graph_) cudaGraphDestroy(graph_);
    graphExec_ = nullptr;
    graph_ = nullptr;
}

void PostProcessor::reallocate(int width, int height)
{
    destroyGraph();
    cudaFree(bloomA_);
    cudaFree(bloomB_);
    width_ = width;
    height_ = height;
    bw_ = (width + 3) / 4;
    bh_ = (height + 3) / 4;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&bloomA_), sizeof(float4) * bw_ * bh_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&bloomB_), sizeof(float4) * bw_ * bh_));
}

void PostProcessor::run(const void* input, bool inputIsHalf, unsigned char* outputRgba8, bool bgra, int width,
                        int height, const Params& p, cudaStream_t stream)
{
    if (width != width_ || height != height_) reallocate(width, height);
    *hParams_ = p;
    CUDA_CHECK(cudaMemcpyAsync(dParams_, hParams_, sizeof(Params), cudaMemcpyHostToDevice, stream));

    const bool sameGraph = graphExec_ && graphInput_ == input && graphOutput_ == outputRgba8 &&
                           graphHalf_ == inputIsHalf && graphBgra_ == bgra;
    if (!sameGraph) {
        destroyGraph();
        // Capture de la chaîne complète dans un CUDA Graph (un seul lancement par image)
        CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
        const dim3 block(16, 16);
        const dim3 gridB((bw_ + 15) / 16, (bh_ + 15) / 16);
        const dim3 gridF((width + 15) / 16, (height + 15) / 16);
        bloomDownsample<<<gridB, block, 0, stream>>>(input, inputIsHalf, width, height, bloomA_, bw_, bh_, dParams_);
        bloomBlur<<<gridB, block, 0, stream>>>(bloomA_, bloomB_, bw_, bh_, 1, 0);
        bloomBlur<<<gridB, block, 0, stream>>>(bloomB_, bloomA_, bw_, bh_, 0, 1);
        composite<<<gridF, block, 0, stream>>>(input, inputIsHalf, bloomA_, bw_, bh_,
                                               reinterpret_cast<uchar4*>(outputRgba8), width, height, bgra, dParams_);
        CUDA_CHECK(cudaStreamEndCapture(stream, &graph_));
        CUDA_CHECK(cudaGraphInstantiate(&graphExec_, graph_, 0ull));
        graphInput_ = input;
        graphOutput_ = outputRgba8;
        graphHalf_ = inputIsHalf;
        graphBgra_ = bgra;
    }
    CUDA_CHECK(cudaGraphLaunch(graphExec_, stream));
}

}  // namespace crtx
