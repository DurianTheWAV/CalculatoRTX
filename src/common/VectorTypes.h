// CalculatoRTX - types vectoriels float2 / float3 / float4 / uint2 / uint3 / uchar4.
//
// Avec CUDA (backend NVIDIA), ce sont ceux du CUDA Toolkit. Sans CUDA (build AMD /
// Vulkan pur), des équivalents binaires compatibles (même taille, même alignement) sont
// définis ici : tout le code hôte (scène, maillages, police) reste identique.
#pragma once

#if defined(__CUDACC__) || defined(CRTX_WITH_CUDA)

#include <vector_functions.h>
#include <vector_types.h>

#else

struct alignas(8) float2 {
    float x, y;
};
struct float3 {
    float x, y, z;
};
struct alignas(16) float4 {
    float x, y, z, w;
};
struct alignas(8) uint2 {
    unsigned int x, y;
};
struct uint3 {
    unsigned int x, y, z;
};
struct alignas(4) uchar4 {
    unsigned char x, y, z, w;
};

inline float2 make_float2(float x, float y) { return float2{x, y}; }
inline float3 make_float3(float x, float y, float z) { return float3{x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return float4{x, y, z, w}; }
inline uint2 make_uint2(unsigned int x, unsigned int y) { return uint2{x, y}; }
inline uint3 make_uint3(unsigned int x, unsigned int y, unsigned int z) { return uint3{x, y, z}; }
inline uchar4 make_uchar4(unsigned char x, unsigned char y, unsigned char z, unsigned char w)
{
    return uchar4{x, y, z, w};
}

#endif

static_assert(sizeof(float2) == 8 && sizeof(float3) == 12 && sizeof(float4) == 16, "types vectoriels");
static_assert(sizeof(uint3) == 12, "types vectoriels");
