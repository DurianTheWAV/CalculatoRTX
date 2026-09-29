// CalculatoRTX - moteur de rendu OptiX (RT cores) + débruiteur IA (Tensor cores).
#pragma once

#include "../scene/CalculatorScene.h"
#include "LaunchParams.h"

#include <cuda_runtime.h>
#include <optix.h>

#include <string>
#include <vector>

namespace crtx {

struct RenderSettings {
    unsigned spp = 1;
    unsigned maxBounces = 5;
    bool ser = true;
    bool denoise = true;
    bool fireflyClamp = true;
};

struct FrameInput {
    CameraData camera;
    CameraData prevCamera;
    float2 jitter = make_float2(0, 0);
    bool dlss = false;
    bool resetAccumulation = false;
    unsigned frameIndex = 0;
    float time = 0.0f;
};

class OptixRenderer {
public:
    OptixRenderer();
    ~OptixRenderer();
    OptixRenderer(const OptixRenderer&) = delete;
    OptixRenderer& operator=(const OptixRenderer&) = delete;

    void init(const std::vector<unsigned char>& ptx);

    // Synchronise la scène (maillages marqués "dirty", instances, matériaux, lumières).
    void syncScene(CalculatorScene& scene, cudaStream_t stream);

    void resize(unsigned width, unsigned height);
    unsigned width() const { return width_; }
    unsigned height() const { return height_; }

    // Lance le path tracing puis (optionnellement) le débruiteur. Retourne le buffer
    // couleur final (float4, résolution de rendu).
    const float4* render(const FrameInput& in, const RenderSettings& rs, cudaStream_t stream);

    // Lance un rayon de sélection ; le résultat est copié en mémoire hôte (lu à l'image suivante).
    void pick(const CameraData& cam, float2 pixel, cudaStream_t stream);
    int pickResult() const { return *hostPick_; }

    const float* depthBuffer() const { return depth_; }
    const float2* motionBuffer() const { return motion_; }

    // Chronomètres GPU (ms) de la dernière image
    float lastTraceMs() const { return traceMs_; }
    float lastDenoiseMs() const { return denoiseMs_; }
    unsigned accumulatedFrames() const { return accumCount_; }

private:
    struct Gas {
        CUdeviceptr buffer = 0;
        OptixTraversableHandle handle = 0;
        CUdeviceptr positions = 0, normals = 0, uvs = 0, indices = 0;
        CUdeviceptr ommArray = 0;
        size_t triangles = 0;
        bool alpha = false;
    };

    void createContext();
    void createModuleAndPipeline(const std::vector<unsigned char>& ptx);
    void createSbt();
    void buildGas(Gas& gas, const SceneMesh& mesh, cudaStream_t stream);
    void freeGas(Gas& gas);
    void buildOpacityMicromap(Gas& gas, const Mesh& mesh, cudaStream_t stream);
    void buildIas(const CalculatorScene& scene, cudaStream_t stream);
    void setupDenoiser();
    void destroyDenoiser();
    void freeFrameBuffers();
    void createEnvironment();
    void applyL2Persistence(cudaStream_t stream);

    OptixDeviceContext context_ = nullptr;
    OptixModule module_ = nullptr;
    OptixPipeline pipeline_ = nullptr;
    OptixProgramGroup pgRaygen_ = nullptr, pgPick_ = nullptr, pgMiss_ = nullptr;
    OptixProgramGroup pgHitOpaque_ = nullptr, pgHitCutout_ = nullptr;
    OptixShaderBindingTable sbt_{};
    OptixShaderBindingTable sbtPick_{};
    CUdeviceptr sbtBuffer_ = 0;

    std::vector<Gas> gas_;
    CUdeviceptr iasBuffer_ = 0, iasTemp_ = 0, instanceBuffer_ = 0;
    size_t iasBufferSize_ = 0, iasTempSize_ = 0, instanceBufferSize_ = 0;
    OptixTraversableHandle ias_ = 0;

    // données de scène côté GPU
    Material* dMaterials_ = nullptr;
    GeometryData* dGeometries_ = nullptr;
    InstanceData* dInstances_ = nullptr;
    RectLight* dLights_ = nullptr;
    size_t materialCap_ = 0, geometryCap_ = 0, instanceCap_ = 0, lightCap_ = 0;
    unsigned numLights_ = 0;
    std::vector<Affine> prevTransforms_;

    // images (résolution de rendu)
    unsigned width_ = 0, height_ = 0;
    float4* frameBlock_ = nullptr;  // color | albedo | normal | accum | denoised (bloc contigu)
    float4 *color_ = nullptr, *albedo_ = nullptr, *normal_ = nullptr, *accum_ = nullptr, *denoised_ = nullptr;
    float* depth_ = nullptr;
    float2* motion_ = nullptr;
    unsigned accumCount_ = 0;

    // débruiteur IA OptiX (réseau de neurones exécuté sur les Tensor cores)
    OptixDenoiser denoiser_ = nullptr;
    CUdeviceptr denoiserState_ = 0, denoiserScratch_ = 0, denoiserIntensity_ = 0;
    size_t denoiserStateSize_ = 0, denoiserScratchSize_ = 0;

    // environnement (texture HDR latitude-longitude générée sur GPU)
    cudaArray_t envArray_ = nullptr;
    cudaTextureObject_t envTex_ = 0;

    LaunchParams* dParams_ = nullptr;
    LaunchParams* hParams_ = nullptr;  // pinned
    int* dPick_ = nullptr;
    int* hostPick_ = nullptr;          // pinned

    cudaEvent_t evTrace0_ = nullptr, evTrace1_ = nullptr, evDenoise1_ = nullptr;
    float traceMs_ = 0.0f, denoiseMs_ = 0.0f;
    bool timingPending_ = false;
    bool l2Configured_ = false;
};

}  // namespace crtx
