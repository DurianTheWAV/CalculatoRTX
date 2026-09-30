// CalculatoRTX - paramètres partagés entre l'hôte et les programmes OptiX (backend NVIDIA).
#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include "SceneTypes.h"

namespace crtx {

struct GeometryData {
    const float3* positions;
    const float3* normals;  // nullptr => ombrage plat
    const float2* uvs;      // peut être nullptr
    const uint3* indices;
};

enum LaunchFlags : unsigned {
    kFlagSER = 1u << 0,          // Shader Execution Reordering (Ada)
    kFlagAccumulate = 1u << 1,   // accumulation progressive
    kFlagDLSS = 1u << 2,         // jitter DLSS
    kFlagFireflyClamp = 1u << 3,
};

struct LaunchParams {
    // ---- sorties (résolution de rendu)
    float4* color;
    float4* albedo;
    float4* normal;
    float* depth;
    float2* motion;
    float4* accum;
    uint2 size;

    unsigned int frameIndex;
    unsigned int accumCount;
    unsigned int spp;
    unsigned int maxBounces;
    unsigned int flags;
    float2 jitter;  // décalage du contenu en pixels (convention DLSS)

    CameraData cam;
    CameraData prevCam;

    OptixTraversableHandle handle;
    const Material* materials;
    const GeometryData* geometries;
    const InstanceData* instances;
    const RectLight* lights;
    unsigned int numLights;

    cudaTextureObject_t envTex;
    float envIntensity;

    // ---- sélection par rayon (clic souris)
    float2 pickPixel;
    int* pickResult;

    float time;
};

}  // namespace crtx
