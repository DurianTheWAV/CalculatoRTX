// CalculatoRTX - kernels CUDA hors OptiX (environnement, conversion, post-traitement).
#pragma once

#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace crtx {

namespace kernels {

// Génère l'environnement HDR (studio photo) en projection latitude-longitude.
void bakeEnvironment(float4* out, int width, int height, cudaStream_t stream);

// Conversion float4 -> RGBA16F (entrée couleur du DLSS), NaN/Inf neutralisés.
void packHalf4(const float4* in, __half* out, int count, cudaStream_t stream);

}  // namespace kernels

// Chaîne de post-traitement exécutée en résolution d'affichage :
// bloom (seuil + flou gaussien séparable) -> exposition -> tone mapping ACES ->
// vignettage -> encodage sRGB + tramage -> BGRA8/RGBA8 pour la swapchain Vulkan.
// Les kernels sont capturés dans un CUDA Graph et rejoués à chaque image.
class PostProcessor {
public:
    PostProcessor();
    ~PostProcessor();
    PostProcessor(const PostProcessor&) = delete;
    PostProcessor& operator=(const PostProcessor&) = delete;

    struct Params {
        float exposure = 1.0f;
        float bloomStrength = 0.06f;
        float bloomThreshold = 1.2f;
        float vignette = 0.22f;
        unsigned frame = 0;
    };

    // input : float4 (inputIsHalf=false) ou RGBA16F (inputIsHalf=true), largeur x hauteur
    void run(const void* input, bool inputIsHalf, unsigned char* outputRgba8, bool bgra, int width, int height,
             const Params& p, cudaStream_t stream);

private:
    void reallocate(int width, int height);
    void destroyGraph();

    int width_ = 0, height_ = 0;
    int bw_ = 0, bh_ = 0;
    float4* bloomA_ = nullptr;
    float4* bloomB_ = nullptr;
    Params* dParams_ = nullptr;
    Params* hParams_ = nullptr;  // pinned

    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t graphExec_ = nullptr;
    const void* graphInput_ = nullptr;
    unsigned char* graphOutput_ = nullptr;
    bool graphHalf_ = false;
    bool graphBgra_ = false;
};

}  // namespace crtx
