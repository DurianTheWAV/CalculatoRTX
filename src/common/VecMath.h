// CalculatoRTX - mathématiques vectorielles partagées hôte / GPU
#pragma once

#include "VectorTypes.h"

#include <math.h>

#include <cmath>

#if defined(__CUDACC__)
#define CRTX_HD __host__ __device__ __forceinline__
#else
#define CRTX_HD inline
#endif

namespace crtx {
constexpr float kPi = 3.14159265358979323846f;
constexpr float kTwoPi = 6.28318530717958647692f;
constexpr float kInvPi = 0.31830988618379067154f;
}  // namespace crtx

// ---------------------------------------------------------------- float2
CRTX_HD float2 operator+(float2 a, float2 b) { return make_float2(a.x + b.x, a.y + b.y); }
CRTX_HD float2 operator-(float2 a, float2 b) { return make_float2(a.x - b.x, a.y - b.y); }
CRTX_HD float2 operator*(float2 a, float s) { return make_float2(a.x * s, a.y * s); }
CRTX_HD float2 operator*(float s, float2 a) { return make_float2(a.x * s, a.y * s); }
CRTX_HD float2 operator*(float2 a, float2 b) { return make_float2(a.x * b.x, a.y * b.y); }

// ---------------------------------------------------------------- float3
CRTX_HD float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }
CRTX_HD float3 operator-(float3 a, float3 b) { return make_float3(a.x - b.x, a.y - b.y, a.z - b.z); }
CRTX_HD float3 operator-(float3 a) { return make_float3(-a.x, -a.y, -a.z); }
CRTX_HD float3 operator*(float3 a, float3 b) { return make_float3(a.x * b.x, a.y * b.y, a.z * b.z); }
CRTX_HD float3 operator*(float3 a, float s) { return make_float3(a.x * s, a.y * s, a.z * s); }
CRTX_HD float3 operator*(float s, float3 a) { return make_float3(a.x * s, a.y * s, a.z * s); }
CRTX_HD float3 operator/(float3 a, float s) { return a * (1.0f / s); }
CRTX_HD float3 operator/(float3 a, float3 b) { return make_float3(a.x / b.x, a.y / b.y, a.z / b.z); }
CRTX_HD float3& operator+=(float3& a, float3 b) { a = a + b; return a; }
CRTX_HD float3& operator-=(float3& a, float3 b) { a = a - b; return a; }
CRTX_HD float3& operator*=(float3& a, float3 b) { a = a * b; return a; }
CRTX_HD float3& operator*=(float3& a, float s) { a = a * s; return a; }

CRTX_HD float dot(float3 a, float3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
CRTX_HD float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
CRTX_HD float length(float3 a) { return sqrtf(dot(a, a)); }
CRTX_HD float3 normalize(float3 a)
{
    const float l = length(a);
    return l > 0.0f ? a * (1.0f / l) : make_float3(0.0f, 0.0f, 1.0f);
}
CRTX_HD float3 lerp(float3 a, float3 b, float t) { return a + (b - a) * t; }
CRTX_HD float lerpf(float a, float b, float t) { return a + (b - a) * t; }
CRTX_HD float clampf(float v, float lo, float hi) { return v < lo ? lo : (v > hi ? hi : v); }
CRTX_HD float saturate(float v) { return clampf(v, 0.0f, 1.0f); }
CRTX_HD float3 fminf3(float3 a, float3 b) { return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z)); }
CRTX_HD float3 fmaxf3(float3 a, float3 b) { return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z)); }
CRTX_HD float maxComp(float3 a) { return fmaxf(a.x, fmaxf(a.y, a.z)); }
CRTX_HD float luminance(float3 c) { return 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z; }
CRTX_HD float3 reflect(float3 i, float3 n) { return i - n * (2.0f * dot(i, n)); }
CRTX_HD float smoothstepf(float e0, float e1, float x)
{
    const float t = saturate((x - e0) / (e1 - e0));
    return t * t * (3.0f - 2.0f * t);
}

// ---------------------------------------------------------------- float4
CRTX_HD float4 make_float4v(float3 a, float w) { return make_float4(a.x, a.y, a.z, w); }
CRTX_HD float3 xyz(float4 a) { return make_float3(a.x, a.y, a.z); }
CRTX_HD float4 operator+(float4 a, float4 b) { return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
CRTX_HD float4 operator*(float4 a, float s) { return make_float4(a.x * s, a.y * s, a.z * s, a.w * s); }

// ---------------------------------------------------------------- Affine 3x4 (format OptiX, lignes)
// m[0..3] = ligne X, m[4..7] = ligne Y, m[8..11] = ligne Z ; point' = M * (p,1)
struct Affine {
    float m[12];
};

CRTX_HD Affine affineIdentity()
{
    Affine a{};
    a.m[0] = a.m[5] = a.m[10] = 1.0f;
    return a;
}

CRTX_HD float3 xformPoint(const float* m, float3 p)
{
    return make_float3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3],
                       m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                       m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

CRTX_HD float3 xformVector(const float* m, float3 v)
{
    return make_float3(m[0] * v.x + m[1] * v.y + m[2] * v.z,
                       m[4] * v.x + m[5] * v.y + m[6] * v.z,
                       m[8] * v.x + m[9] * v.y + m[10] * v.z);
}

CRTX_HD Affine affineMul(const Affine& a, const Affine& b)
{
    Affine r{};
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 4; ++j) {
            float v = a.m[i * 4 + 0] * b.m[0 * 4 + j] + a.m[i * 4 + 1] * b.m[1 * 4 + j] +
                      a.m[i * 4 + 2] * b.m[2 * 4 + j];
            if (j == 3) v += a.m[i * 4 + 3];
            r.m[i * 4 + j] = v;
        }
    }
    return r;
}

CRTX_HD Affine affineTranslate(float3 t)
{
    Affine a = affineIdentity();
    a.m[3] = t.x;
    a.m[7] = t.y;
    a.m[11] = t.z;
    return a;
}

CRTX_HD Affine affineScale(float3 s)
{
    Affine a{};
    a.m[0] = s.x;
    a.m[5] = s.y;
    a.m[10] = s.z;
    return a;
}

// Rotation autour de l'axe X (radians)
CRTX_HD Affine affineRotateX(float r)
{
    Affine a = affineIdentity();
    const float c = cosf(r), s = sinf(r);
    a.m[5] = c;
    a.m[6] = -s;
    a.m[9] = s;
    a.m[10] = c;
    return a;
}

CRTX_HD Affine affineRotateY(float r)
{
    Affine a = affineIdentity();
    const float c = cosf(r), s = sinf(r);
    a.m[0] = c;
    a.m[2] = s;
    a.m[8] = -s;
    a.m[10] = c;
    return a;
}
