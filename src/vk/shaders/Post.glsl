// CalculatoRTX - déclarations communes des passes de post-traitement (backend Vulkan).
// Chaque passe reçoit ses tampons par adresse GPU dans les constantes de poussée.
#ifndef CRTX_POST_GLSL
#define CRTX_POST_GLSL

#extension GL_EXT_buffer_reference : require
#extension GL_EXT_scalar_block_layout : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require

layout(buffer_reference, scalar, buffer_reference_align = 16) buffer PVec4 { vec4 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer PUint { uint v[]; };

// Miroir de VkPostPush (src/vk/VkRenderer.cpp)
layout(push_constant, scalar) uniform Push {
    uint64_t src;
    uint64_t dst;
    uint64_t aux0;
    uint64_t aux1;
    uint srcW, srcH, dstW, dstH;
    float p0, p1, p2, p3;
    uint u0, u1;
} pc;

float luminanceP(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

#endif
