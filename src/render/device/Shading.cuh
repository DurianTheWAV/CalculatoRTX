// CalculatoRTX - fonctions de shading GPU : aléatoire, bruit, motifs procéduraux, BSDF.
#pragma once

#include "../LaunchParams.h"

namespace crtx {
namespace dev {

// ============================================================ nombres aléatoires (PCG)
__device__ __forceinline__ unsigned pcgHash(unsigned v)
{
    const unsigned state = v * 747796405u + 2891336453u;
    const unsigned word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

struct Rng {
    unsigned s;
    __device__ float next()
    {
        s = pcgHash(s);
        return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }
    __device__ float2 next2()
    {
        const float a = next();
        return make_float2(a, next());
    }
};

// ============================================================ bruit de valeur 3D
__device__ __forceinline__ float hash3i(int x, int y, int z)
{
    const unsigned h = pcgHash(static_cast<unsigned>(x) * 73856093u ^ pcgHash(static_cast<unsigned>(y) * 19349663u ^
                                                                            static_cast<unsigned>(z) * 83492791u));
    return static_cast<float>(h & 0xFFFFFFu) * (1.0f / 16777215.0f);
}

__device__ float valueNoise(float3 p)
{
    const float fx = floorf(p.x), fy = floorf(p.y), fz = floorf(p.z);
    const int ix = static_cast<int>(fx), iy = static_cast<int>(fy), iz = static_cast<int>(fz);
    float tx = p.x - fx, ty = p.y - fy, tz = p.z - fz;
    tx = tx * tx * (3.0f - 2.0f * tx);
    ty = ty * ty * (3.0f - 2.0f * ty);
    tz = tz * tz * (3.0f - 2.0f * tz);
    const float c000 = hash3i(ix, iy, iz), c100 = hash3i(ix + 1, iy, iz);
    const float c010 = hash3i(ix, iy + 1, iz), c110 = hash3i(ix + 1, iy + 1, iz);
    const float c001 = hash3i(ix, iy, iz + 1), c101 = hash3i(ix + 1, iy, iz + 1);
    const float c011 = hash3i(ix, iy + 1, iz + 1), c111 = hash3i(ix + 1, iy + 1, iz + 1);
    const float x00 = lerpf(c000, c100, tx), x10 = lerpf(c010, c110, tx);
    const float x01 = lerpf(c001, c101, tx), x11 = lerpf(c011, c111, tx);
    return lerpf(lerpf(x00, x10, ty), lerpf(x01, x11, ty), tz);
}

__device__ float fbm(float3 p, int octaves)
{
    float a = 0.5f, s = 0.0f;
    for (int i = 0; i < octaves; ++i) {
        s += a * valueNoise(p);
        p = p * 2.03f + make_float3(17.1f, 3.7f, 9.2f);
        a *= 0.5f;
    }
    return s;
}

__device__ __forceinline__ float fractf(float x) { return x - floorf(x); }

// ============================================================ matériau évalué au point d'impact
struct Surface {
    float3 base;
    float3 emission;
    float roughness;
    float metallic;
    float transmission;
    float ior;
    float clearcoat;
    float ccRoughness;
    float specular;
};

// P : position dans le repère de l'objet (les motifs suivent l'objet s'il est incliné ou animé)
__device__ Surface evalMaterial(const Material& m, float3 P, float2 uv)
{
    Surface s;
    s.base = m.baseColor;
    s.emission = m.emission;
    s.roughness = m.roughness;
    s.metallic = m.metallic;
    s.transmission = m.transmission;
    s.ior = m.ior;
    s.clearcoat = m.clearcoat;
    s.ccRoughness = m.clearcoatRoughness;
    s.specular = m.specular;

    switch (m.pattern) {
        case kPatternWood: {
            const float3 q = P * m.patternScale;
            const float plank = floorf(q.z / 3.4f);
            const float h = hash3i(static_cast<int>(plank), 7, 11);
            const float w = fbm(make_float3(q.x * 0.12f + h * 31.0f, q.z * 1.1f, h * 5.0f), 4);
            const float grain = q.z * 5.0f + w * 3.2f + sinf(q.x * 0.21f + h * 6.28f) * 0.9f;
            float rings = 0.5f + 0.5f * sinf(grain * 3.1f);
            rings = rings * rings;
            const float fine = valueNoise(make_float3(q.x * 0.6f, q.z * 40.0f, h * 3.0f));
            const float3 dark = make_float3(0.055f, 0.026f, 0.012f);
            const float3 light = make_float3(0.24f, 0.12f, 0.055f);
            s.base = lerp(dark, light, 0.25f + 0.6f * rings + 0.15f * fine) * (0.8f + 0.35f * h);
            const float seam = fractf(q.z / 3.4f);
            if (seam < 0.004f || seam > 0.996f) s.base = s.base * 0.25f;
            s.roughness = m.roughness * (0.8f + 0.5f * fine);
            break;
        }
        case kPatternBrushed: {
            const float n = valueNoise(make_float3(P.x * 1.5f, P.z * 220.0f, P.y * 220.0f));
            const float n2 = valueNoise(make_float3(P.x * 0.3f, P.z * 40.0f, 3.0f));
            s.base = s.base * (0.9f + 0.14f * n + 0.05f * n2);
            s.roughness = m.roughness * (0.75f + 0.5f * n);
            break;
        }
        case kPatternSolar: {
            const float fu = fractf(uv.x * m.patternScale);
            const float fv = fractf(uv.y * 7.0f);
            const float n = valueNoise(make_float3(uv.x * 60.0f, uv.y * 12.0f, 1.0f));
            s.base = make_float3(0.03f, 0.018f, 0.045f) * (0.7f + 0.6f * n);
            if (fu < 0.015f || fu > 0.985f) {
                s.base = make_float3(0.55f, 0.55f, 0.58f);
                s.metallic = 1.0f;
                s.roughness = 0.3f;
            } else if (fv < 0.05f) {
                s.base = make_float3(0.35f, 0.35f, 0.38f);
                s.metallic = 1.0f;
                s.roughness = 0.35f;
            }
            break;
        }
        case kPatternVfd: {
            const float fx = fractf(uv.x * 154.0f), fy = fractf(uv.y * 33.0f);
            if (fx < 0.07f || fy < 0.07f) s.base = make_float3(0.03f, 0.07f, 0.065f);
            break;
        }
        case kPatternCarbon: {
            const float sc = m.patternScale;
            const float cx = floorf(P.x * sc), cz = floorf(P.z * sc);
            const bool warp = (static_cast<int>(cx + cz) & 1) != 0;
            const float t = warp ? fractf(P.x * sc) : fractf(P.z * sc);
            const float sheen = 0.55f + 0.9f * sinf(t * kPi);
            s.base = s.base * sheen;
            s.roughness = m.roughness * (warp ? 0.8f : 1.25f);
            break;
        }
        case kPatternPcb: {
            // Vernis épargne vert ; pistes de cuivre visibles en plus clair sous le vernis,
            // routées par zones de 1.3 cm orientées en X ou en Z ; vias étamés.
            const float rx = floorf(P.x / 1.3f), rz = floorf(P.z / 1.3f);
            const bool alongX = hash3i(static_cast<int>(rx), static_cast<int>(rz), 3) < 0.5f;
            const float along = alongX ? P.x : P.z, across = alongX ? P.z : P.x;
            const float lane = floorf(across / 0.1f), fl = fractf(across / 0.1f);
            const float laneSeed = hash3i(static_cast<int>(lane), 5, static_cast<int>(alongX ? rz : rx));
            const float seg = floorf(along / 0.7f + laneSeed);
            const float hs = hash3i(static_cast<int>(lane), static_cast<int>(seg), static_cast<int>(rx * 7.0f + rz * 13.0f));
            if (hs < 0.55f && fabsf(fl - 0.5f) < 0.2f) s.base = s.base * 2.3f + make_float3(0.01f, 0.03f, 0.0f);
            const float vx = P.x / 0.45f, vz = P.z / 0.45f;
            const float hv = hash3i(static_cast<int>(floorf(vx)), static_cast<int>(floorf(vz)), 9);
            const float dx = fractf(vx) - 0.5f, dz = fractf(vz) - 0.5f;
            const float r = sqrtf(dx * dx + dz * dz) * 0.45f;
            if (hv < 0.18f && r < 0.04f) {
                if (r < 0.017f) {
                    s.base = make_float3(0.01f, 0.01f, 0.01f);
                    s.roughness = 0.8f;
                } else {
                    s.base = make_float3(0.78f, 0.78f, 0.8f);
                    s.metallic = 1.0f;
                    s.roughness = 0.3f;
                }
                s.clearcoat = 0.0f;
            }
            break;
        }
        default:
            break;
    }
    return s;
}

// ============================================================ BSDF (Lambert + GGX + vernis)
struct Onb {
    float3 t, b, n;
};

__device__ __forceinline__ Onb makeOnb(float3 n)
{
    // Duff et al. 2017 (repère orthonormé sans branche)
    const float sign = copysignf(1.0f, n.z);
    const float a = -1.0f / (sign + n.z);
    const float b = n.x * n.y * a;
    Onb o;
    o.t = make_float3(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    o.b = make_float3(b, sign + n.y * n.y * a, -n.y);
    o.n = n;
    return o;
}

__device__ __forceinline__ float3 toLocal(const Onb& o, float3 v) { return make_float3(dot(v, o.t), dot(v, o.b), dot(v, o.n)); }
__device__ __forceinline__ float3 toWorld(const Onb& o, float3 v) { return o.t * v.x + o.b * v.y + o.n * v.z; }

__device__ __forceinline__ float3 schlick3(float3 f0, float c)
{
    const float t = powf(fmaxf(1.0f - c, 0.0f), 5.0f);
    return f0 + (make_float3(1, 1, 1) - f0) * t;
}
__device__ __forceinline__ float schlick1(float f0, float c)
{
    const float t = powf(fmaxf(1.0f - c, 0.0f), 5.0f);
    return f0 + (1.0f - f0) * t;
}

__device__ __forceinline__ float ggxD(float noh, float a2)
{
    const float d = noh * noh * (a2 - 1.0f) + 1.0f;
    return a2 / (kPi * d * d + 1e-20f);
}

__device__ __forceinline__ float smithG1(float nov, float a2)
{
    return 2.0f * nov / (nov + sqrtf(a2 + (1.0f - a2) * nov * nov) + 1e-20f);
}

// Échantillonnage des normales visibles (Heitz 2018)
__device__ float3 sampleVndf(float3 v, float a, float2 u)
{
    const float3 vh = normalize(make_float3(a * v.x, a * v.y, v.z));
    const float lensq = vh.x * vh.x + vh.y * vh.y;
    const float3 t1 = lensq > 0.0f ? make_float3(-vh.y, vh.x, 0.0f) * rsqrtf(lensq) : make_float3(1, 0, 0);
    const float3 t2 = cross(vh, t1);
    const float r = sqrtf(u.x);
    const float phi = kTwoPi * u.y;
    const float p1 = r * cosf(phi);
    float p2 = r * sinf(phi);
    const float s = 0.5f * (1.0f + vh.z);
    p2 = (1.0f - s) * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1)) + s * p2;
    const float3 nh = t1 * p1 + t2 * p2 + vh * sqrtf(fmaxf(0.0f, 1.0f - p1 * p1 - p2 * p2));
    return normalize(make_float3(a * nh.x, a * nh.y, fmaxf(1e-6f, nh.z)));
}

struct BsdfLobes {
    float3 f0;
    float3 diffuse;
    float a2Spec, aSpec;
    float a2Coat, aCoat;
    float coat;
    float pDiff, pSpec, pCoat;
};

__device__ BsdfLobes makeLobes(const Surface& s, float nov)
{
    BsdfLobes l;
    const float f0d = 0.08f * s.specular;
    l.f0 = lerp(make_float3(f0d, f0d, f0d), s.base, s.metallic);
    l.diffuse = s.base * (1.0f - s.metallic);
    l.aSpec = fmaxf(s.roughness * s.roughness, 1e-3f);
    l.a2Spec = l.aSpec * l.aSpec;
    l.aCoat = fmaxf(s.ccRoughness * s.ccRoughness, 1e-3f);
    l.a2Coat = l.aCoat * l.aCoat;
    l.coat = s.clearcoat;
    const float es = luminance(schlick3(l.f0, nov));
    const float ed = luminance(l.diffuse) * (1.0f - es);
    const float ec = l.coat * schlick1(0.04f, nov);
    const float sum = es + ed + ec;
    if (sum <= 0.0f) {
        l.pDiff = 1.0f;
        l.pSpec = l.pCoat = 0.0f;
    } else {
        l.pDiff = ed / sum;
        l.pSpec = es / sum;
        l.pCoat = ec / sum;
    }
    return l;
}

// wo, wi en repère local (z = normale)
__device__ float3 evalBsdf(const BsdfLobes& l, float3 wo, float3 wi, float& pdf)
{
    pdf = 0.0f;
    const float nov = wo.z, nol = wi.z;
    if (nov <= 0.0f || nol <= 0.0f) return make_float3(0, 0, 0);
    const float3 h = normalize(wo + wi);
    const float noh = fmaxf(h.z, 0.0f), voh = fmaxf(dot(wo, h), 0.0f);

    const float3 fs = schlick3(l.f0, voh) * (ggxD(noh, l.a2Spec) * smithG1(nov, l.a2Spec) * smithG1(nol, l.a2Spec) /
                                              (4.0f * nov * nol));
    const float3 fd = l.diffuse * (kInvPi * (1.0f - luminance(schlick3(l.f0, nov))));
    const float fc = l.coat * schlick1(0.04f, voh);
    const float coatSpec = fc * ggxD(noh, l.a2Coat) * smithG1(nov, l.a2Coat) * smithG1(nol, l.a2Coat) / (4.0f * nov * nol);

    pdf = l.pDiff * nol * kInvPi +
          l.pSpec * ggxD(noh, l.a2Spec) * smithG1(nov, l.a2Spec) / (4.0f * nov) +
          l.pCoat * ggxD(noh, l.a2Coat) * smithG1(nov, l.a2Coat) / (4.0f * nov);
    return (fd + fs) * (1.0f - fc) + make_float3(coatSpec, coatSpec, coatSpec);
}

__device__ bool sampleBsdf(const BsdfLobes& l, float3 wo, Rng& rng, float3& wi, float3& f, float& pdf)
{
    const float u = rng.next();
    const float2 u2 = rng.next2();
    if (u < l.pDiff) {
        const float r = sqrtf(u2.x), phi = kTwoPi * u2.y;
        wi = make_float3(r * cosf(phi), r * sinf(phi), sqrtf(fmaxf(0.0f, 1.0f - u2.x)));
    } else {
        const float a = (u < l.pDiff + l.pSpec) ? l.aSpec : l.aCoat;
        const float3 h = sampleVndf(wo, a, u2);
        wi = reflect(wo * -1.0f, h);
    }
    if (wi.z <= 0.0f) return false;
    f = evalBsdf(l, wo, wi, pdf);
    return pdf > 1e-8f;
}

// Fresnel diélectrique exact ; eta = n_incident / n_transmis
__device__ float fresnelDielectric(float cosi, float eta)
{
    const float sint2 = eta * eta * (1.0f - cosi * cosi);
    if (sint2 >= 1.0f) return 1.0f;
    const float cost = sqrtf(1.0f - sint2);
    const float rs = (eta * cosi - cost) / (eta * cosi + cost);
    const float rp = (cosi - eta * cost) / (cosi + eta * cost);
    return 0.5f * (rs * rs + rp * rp);
}

// Transmittance de Beer-Lambert : couleur atteinte après 'ratio' fois la distance de référence
__device__ __forceinline__ float3 beerLambert(float3 colorAtDistance, float ratio)
{
    return make_float3(powf(fmaxf(colorAtDistance.x, 1e-4f), ratio), powf(fmaxf(colorAtDistance.y, 1e-4f), ratio),
                       powf(fmaxf(colorAtDistance.z, 1e-4f), ratio));
}

// Interface diélectrique lisse (roughness = 0) ou dépolie : microfacettes GGX (Walter et al.
// 2007) avec échantillonnage des normales visibles, puis réflexion ou réfraction choisie
// selon Fresnel. nOut : normale extérieure du volume (le sens d'entrée / sortie en découle).
// Retourne false si le chemin s'éteint (direction sous la surface).
__device__ bool sampleDielectric(float3 rd, float3 nOut, float ior, float roughness, Rng& rng, float3& nd,
                                 float& weight, bool& refracted)
{
    const bool entering = dot(rd, nOut) < 0.0f;
    const float3 n = entering ? nOut : nOut * -1.0f;  // normale côté incident
    const float eta = entering ? 1.0f / ior : ior;
    float3 m = n;
    float a2 = 0.0f;
    const float2 u2 = rng.next2();
    if (roughness > 0.02f) {
        const float a = fmaxf(roughness * roughness, 1e-3f);
        a2 = a * a;
        const Onb onb = makeOnb(n);
        float3 wo = toLocal(onb, rd * -1.0f);
        wo.z = fmaxf(wo.z, 1e-4f);
        m = toWorld(onb, sampleVndf(normalize(wo), a, u2));
    }
    const float cosi = dot(rd * -1.0f, m);
    if (cosi <= 0.0f) return false;
    const float F = fresnelDielectric(fminf(cosi, 1.0f), eta);
    refracted = rng.next() >= F;
    if (!refracted) {
        nd = reflect(rd, m);
        if (dot(nd, n) <= 0.0f) return false;
    } else {
        const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
        nd = normalize(rd * eta + m * (eta * cosi - sqrtf(fmaxf(k, 0.0f))));
        if (dot(nd, n) >= 0.0f) return false;
    }
    // poids VNDF : terme d'ombrage de Smith de la direction sortante
    weight = a2 > 0.0f ? smithG1(fabsf(dot(nd, n)), a2) : 1.0f;
    return true;
}

}  // namespace dev
}  // namespace crtx
