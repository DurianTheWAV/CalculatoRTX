// CalculatoRTX - moteur de calcul GPU.
//
// Un warp de 32 threads traite une requête :
//   * voie 0      : analyse lexicale + algorithme de Dijkstra (shunting-yard) -> RPN
//                   (les littéraux sont convertis en double-double sur le GPU)
//   * voies 0..3  : évaluation concurrente de la RPN en double-double, FP64, FP32 et
//                   arithmétique d'intervalles FP64 à arrondis dirigés (__dadd_rd/__dadd_ru...)
//   * réduction   : échange des résultats par primitives de warp (__shfl_sync)
//   * voie 0      : formatage décimal du résultat double-double (sur GPU également)
#include "CalcEngine.h"
#include "CalcCore.cuh"

#include "../common/Check.h"

#include <cooperative_groups.h>

#include <cfloat>
#include <cmath>
#include <cstring>

namespace cg = cooperative_groups;

namespace crtx {
namespace calc {

using dd::DD;

// ============================================================================
//                                   Kernel
// ============================================================================
namespace {

using namespace core;

__global__ void __launch_bounds__(32) calcKernel(const Request* __restrict__ req, Result* __restrict__ res)
{
    __shared__ Request rq;
    __shared__ RpnItem rpn[kMaxRpn];
    __shared__ int rpnCount;
    __shared__ int parseStatus;

    cg::thread_block block = cg::this_thread_block();
    cg::thread_block_tile<32> warp = cg::tiled_partition<32>(block);
    const int lane = warp.thread_rank();

    // Copie coopérative de la requête en mémoire partagée
    const int* src = reinterpret_cast<const int*>(req);
    int* dst = reinterpret_cast<int*>(&rq);
    for (int i = lane; i < static_cast<int>(sizeof(Request) / sizeof(int)); i += 32) dst[i] = src[i];
    warp.sync();

    if (lane == 0) {
        int st = kOk;
        rpnCount = parseProgram(rq, rpn, st);
        parseStatus = st;
    }
    warp.sync();

    // Évaluation concurrente dans 4 arithmétiques différentes
    int status = parseStatus;
    DD vDD = dd::make(0.0);
    double v64 = 0.0;
    float v32 = 0.0f;
    Iv vIv{0.0, 0.0};
    if (status == kOk) {
        if (lane == 0) status = evaluateRpn<NumDD>(rpn, rpnCount, rq.angleMode, vDD);
        else if (lane == 1) status = evaluateRpn<NumFP<double>>(rpn, rpnCount, rq.angleMode, v64);
        else if (lane == 2) status = evaluateRpn<NumFP<float>>(rpn, rpnCount, rq.angleMode, v32);
        else if (lane == 3) status = evaluateRpn<NumIv>(rpn, rpnCount, rq.angleMode, vIv);
    }
    warp.sync();

    // Réduction par échanges intra-warp (registres -> registres, sans mémoire partagée)
    const double f64 = warp.shfl(v64, 1);
    const float f32 = warp.shfl(v32, 2);
    const double ivLo = warp.shfl(vIv.lo, 3);
    const double ivHi = warp.shfl(vIv.hi, 3);
    const int st64 = warp.shfl(status, 1);
    const int st32 = warp.shfl(status, 2);
    const int stIv = warp.shfl(status, 3);

    if (lane == 0) {
        Result r{};
        finalizeResult(r, status, rpnCount, vDD, f64, f32, Iv{ivLo, ivHi}, st64, st32, stIv);
        *res = r;
    }
}

}  // namespace

// ============================================================================
//                                  Hôte
// ============================================================================
CalcEngine::CalcEngine()
{
    int leastPriority = 0, greatestPriority = 0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));
    // Flux haute priorité : le calcul passe devant le rendu ray tracing en cours.
    CUDA_CHECK(cudaStreamCreateWithPriority(&stream_, cudaStreamNonBlocking, greatestPriority));
    CUDA_CHECK(cudaEventCreate(&evStart_));
    CUDA_CHECK(cudaEventCreate(&evStop_));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostReq_), sizeof(Request)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostRes_), sizeof(Result)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devReq_), sizeof(Request)));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&devRes_), sizeof(Result)));
}

CalcEngine::~CalcEngine()
{
    cudaFree(devRes_);
    cudaFree(devReq_);
    cudaFreeHost(hostRes_);
    cudaFreeHost(hostReq_);
    cudaEventDestroy(evStop_);
    cudaEventDestroy(evStart_);
    cudaStreamDestroy(stream_);
}

Result CalcEngine::evaluate(const std::string& program, AngleMode mode, double ansHi, double ansLo,
                            double memHi, double memLo, float* gpuMicroseconds)
{
    *hostReq_ = makeRequest(program, mode, ansHi, ansLo, memHi, memLo);

    CUDA_CHECK(cudaMemcpyAsync(devReq_, hostReq_, sizeof(Request), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaEventRecord(evStart_, stream_));
    calcKernel<<<1, 32, 0, stream_>>>(devReq_, devRes_);
    CUDA_CHECK_LAST();
    CUDA_CHECK(cudaEventRecord(evStop_, stream_));
    CUDA_CHECK(cudaMemcpyAsync(hostRes_, devRes_, sizeof(Result), cudaMemcpyDeviceToHost, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    if (gpuMicroseconds) {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, evStart_, evStop_));
        *gpuMicroseconds = ms * 1000.0f;
    }
    return *hostRes_;
}

}  // namespace calc
}  // namespace crtx
