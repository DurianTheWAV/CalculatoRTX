// CalculatoRTX - programmes OptiX (compilés en PTX puis optimisés par le pilote pour les RT cores).
//
// Rendu : path tracing complet (tous les pixels, y compris l'arrière-plan, sont issus de
// rayons) avec
//   * traversée matérielle BVH/triangles sur les RT cores (optixTraverse),
//   * Shader Execution Reordering (optixReorder) : regroupement des threads par matériau
//     avant le shading - accéléré par le matériel sur Ada Lovelace (RTX 40),
//   * Opacity Micromaps + any-hit pour la grille perforée,
//   * éclairage direct par échantillonnage des luminaires + MIS, ombres douces,
//     réflexions GGX, réfraction du verre de l'afficheur, illumination globale,
//   * sorties auxiliaires pour DLSS (profondeur, vecteurs de mouvement) et pour le
//     débruiteur IA OptiX (albédo, normales).
#include <optix.h>

#include "../Grille.h"
#include "../LaunchParams.h"
#include "Shading.cuh"

using namespace crtx;
using namespace crtx::dev;

extern "C" {
__constant__ LaunchParams params;
}

namespace {

struct Hit {
    float3 P;        // position monde
    float3 Pobj;     // position objet (pour les vecteurs de mouvement)
    float3 N;        // normale d'ombrage orientée vers le rayon incident
    float3 Ng;       // normale géométrique orientée vers le rayon incident
    float3 Nout;     // normale "extérieure" du maillage (verre : entrée / sortie)
    float2 uv;
    float t;
    int instance;
};

__device__ __forceinline__ float3 cameraRay(const CameraData& c, float2 pix, uint2 size)
{
    const float nx = 2.0f * pix.x / static_cast<float>(size.x) - 1.0f;
    const float ny = 1.0f - 2.0f * pix.y / static_cast<float>(size.y);
    return normalize(c.U * nx + c.V * ny + c.W);
}

__device__ __forceinline__ bool projectToPixel(const CameraData& c, float3 X, uint2 size, float2& pix)
{
    const float3 d = X - c.eye;
    const float z = dot(d, c.W);
    if (z < 1e-4f) return false;
    const float nx = dot(d, c.U) / (dot(c.U, c.U) * z);
    const float ny = dot(d, c.V) / (dot(c.V, c.V) * z);
    pix = make_float2((nx + 1.0f) * 0.5f * size.x, (1.0f - ny) * 0.5f * size.y);
    return true;
}

__device__ __forceinline__ float deviceDepth(const CameraData& c, float3 X)
{
    const float z = fmaxf(dot(X - c.eye, c.W), c.zNear);
    return saturate((c.zFar / (c.zFar - c.zNear)) * (1.0f - c.zNear / z));
}

__device__ __forceinline__ float3 envRadiance(float3 d)
{
    const float u = atan2f(d.x, -d.z) * (0.5f * kInvPi) + 0.5f;
    const float v = acosf(clampf(d.y, -1.0f, 1.0f)) * kInvPi;
    const float4 e = tex2D<float4>(params.envTex, u, v);
    return make_float3(e.x, e.y, e.z) * params.envIntensity;
}

__device__ __forceinline__ float3 transformNormalW2OT(const float* w2o, float3 n)
{
    // normale monde = transposée(inverse) * normale objet
    return make_float3(w2o[0] * n.x + w2o[4] * n.y + w2o[8] * n.z,
                       w2o[1] * n.x + w2o[5] * n.y + w2o[9] * n.z,
                       w2o[2] * n.x + w2o[6] * n.y + w2o[10] * n.z);
}

// Reconstruit le point d'impact à partir de l'objet "hit" SER (sans closest-hit)
__device__ Hit fetchHit(float3 rayDir)
{
    Hit h;
    h.instance = static_cast<int>(optixHitObjectGetInstanceIndex());
    h.t = optixHitObjectGetRayTmax();
    const unsigned prim = optixHitObjectGetPrimitiveIndex();
    const float2 bc = optixHitObjectGetTriangleBarycentrics();
    const InstanceData& inst = params.instances[h.instance];
    const GeometryData& g = params.geometries[inst.geometry];
    const uint3 tri = g.indices[prim];
    const float3 p0 = g.positions[tri.x], p1 = g.positions[tri.y], p2 = g.positions[tri.z];
    const float w0 = 1.0f - bc.x - bc.y;
    h.Pobj = p0 * w0 + p1 * bc.x + p2 * bc.y;

    float o2w[12], w2o[12];
    optixHitObjectGetObjectToWorldTransformMatrix(o2w);
    optixHitObjectGetWorldToObjectTransformMatrix(w2o);
    h.P = xformPoint(o2w, h.Pobj);

    float3 ng = normalize(transformNormalW2OT(w2o, cross(p1 - p0, p2 - p0)));
    float3 ns = ng;
    if (g.normals) {
        const float3 n = g.normals[tri.x] * w0 + g.normals[tri.y] * bc.x + g.normals[tri.z] * bc.y;
        ns = normalize(transformNormalW2OT(w2o, n));
        if (dot(ns, ng) < 0.0f) ng = ng * -1.0f;  // aligne la normale géométrique sur l'extérieur
    }
    h.Nout = ns;
    h.uv = make_float2(0.0f, 0.0f);
    if (g.uvs) {
        const float2 a = g.uvs[tri.x], b = g.uvs[tri.y], c = g.uvs[tri.z];
        h.uv = a * w0 + b * bc.x + c * bc.y;
    }
    // orientation face au rayon
    if (dot(ng, rayDir) > 0.0f) {
        ng = ng * -1.0f;
        ns = ns * -1.0f;
    }
    if (dot(ns, rayDir) > 0.0f) ns = normalize(ns - rayDir * (dot(ns, rayDir) * 1.01f));
    h.Ng = ng;
    h.N = ns;
    return h;
}

__device__ __forceinline__ float3 offsetRay(float3 p, float3 n, float3 dir)
{
    const float eps = 2e-4f * (1.0f + fmaxf(fabsf(p.x), fmaxf(fabsf(p.y), fabsf(p.z))));
    return p + n * (dot(dir, n) > 0.0f ? eps : -eps);
}

__device__ __forceinline__ bool occluded(float3 o, float3 d, float tmax)
{
    optixTraverse(params.handle, o, d, 0.0f, tmax, 0.0f, kMaskSolid,
                  OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT, 0, 1, 0);
    return optixHitObjectIsHit();
}

__device__ __forceinline__ float3 clampLum(float3 c, float maxLum)
{
    const float l = luminance(c);
    return l > maxLum ? c * (maxLum / l) : c;
}

struct PathOutput {
    float3 radiance;
    float3 albedo;
    float3 normal;
    float depth;
    float2 motion;
};

__device__ PathOutput tracePath(float2 pix, Rng& rng, bool firstSample)
{
    const uint2 size = params.size;
    const CameraData& cam = params.cam;
    float3 ro = cam.eye;
    float3 rd = cameraRay(cam, pix, size);

    PathOutput out;
    out.radiance = make_float3(0, 0, 0);
    out.albedo = make_float3(0, 0, 0);
    out.normal = make_float3(0, 0, 0);
    out.depth = 1.0f;
    out.motion = make_float2(0, 0);

    float3 throughput = make_float3(1, 1, 1);
    float lastPdf = 0.0f;
    bool lastDelta = true;  // le rayon primaire compte comme "spéculaire"
    bool guideDone = !firstSample;
    const bool useSer = (params.flags & kFlagSER) != 0;
    const bool clampFire = (params.flags & kFlagFireflyClamp) != 0;

    for (unsigned bounce = 0; bounce <= params.maxBounces; ++bounce) {
        optixTraverse(params.handle, ro, rd, 0.0f, 1e16f, 0.0f, kMaskAll, OPTIX_RAY_FLAG_NONE, 0, 1, 0);

        // ---- Shader Execution Reordering : cohérence par matériau
        if (useSer) {
            unsigned hint = 0xFFu;
            if (optixHitObjectIsHit())
                hint = static_cast<unsigned>(params.instances[optixHitObjectGetInstanceIndex()].material) & 0xFFu;
            optixReorder(hint, 8);
        }

        if (!optixHitObjectIsHit()) {
            const float3 env = envRadiance(rd);
            float3 c = throughput * env;
            if (clampFire && bounce > 0) c = clampLum(c, 20.0f);
            out.radiance += c;
            if (!guideDone) {
                out.albedo = fminf3(env, make_float3(1, 1, 1));
                out.normal = make_float3(0, 0, 0);
                guideDone = true;
            }
            if (bounce == 0 && firstSample) {
                float2 prev;
                if (projectToPixel(params.prevCam, params.prevCam.eye + rd * 1000.0f, size, prev))
                    out.motion = prev - pix;
            }
            break;
        }

        const Hit h = fetchHit(rd);
        const InstanceData& inst = params.instances[h.instance];
        const Material& m = params.materials[inst.material];
        Surface s = evalMaterial(m, h.P, h.uv);
        const bool isGlass = s.transmission > 0.5f;

        if (bounce == 0 && firstSample) {
            out.depth = deviceDepth(cam, h.P);
            float2 prev;
            const float3 prevP = xformPoint(inst.prevObjectToWorld, h.Pobj);
            if (projectToPixel(params.prevCam, prevP, size, prev)) out.motion = prev - pix;
        }
        if (!guideDone && (!isGlass || bounce >= 3)) {
            out.albedo = fminf3(s.base + s.emission, make_float3(1, 1, 1));
            out.normal = h.N;
            guideDone = true;
        }

        // ---- émission (surbrillance des touches incluse)
        float3 Le = s.emission * (1.0f + 1.5f * inst.glow) + inst.glowColor * (inst.glow * 0.5f);
        if (inst.lightIndex >= 0) {
            const RectLight& L = params.lights[inst.lightIndex];
            const float cosL = -dot(rd, L.normal);
            if (cosL <= 0.0f) Le = make_float3(0, 0, 0);
            else if (!lastDelta) {
                const float pdfL = h.t * h.t / (cosL * L.area * params.numLights);
                const float w = lastPdf * lastPdf / (lastPdf * lastPdf + pdfL * pdfL);
                Le = Le * w;
            }
        }
        if (maxComp(Le) > 0.0f) {
            float3 c = throughput * Le;
            if (clampFire && bounce > 0) c = clampLum(c, 30.0f);
            out.radiance += c;
        }
        if (bounce == params.maxBounces) break;
        if (inst.lightIndex >= 0) break;  // les luminaires n'ont pas de réflexion

        // ---- verre lisse : réflexion / réfraction de Fresnel (lobe de Dirac)
        if (isGlass) {
            const bool entering = dot(rd, h.Nout) < 0.0f;
            const float3 n = entering ? h.Nout : h.Nout * -1.0f;
            const float eta = entering ? 1.0f / s.ior : s.ior;
            const float cosi = fminf(1.0f, -dot(rd, n));
            const float F = fresnelDielectric(cosi, eta);
            float3 nd;
            if (rng.next() < F) {
                nd = reflect(rd, n);
            } else {
                const float k = 1.0f - eta * eta * (1.0f - cosi * cosi);
                nd = normalize(rd * eta + n * (eta * cosi - sqrtf(fmaxf(k, 0.0f))));
                throughput *= s.base;
            }
            ro = offsetRay(h.P, n, nd);
            rd = nd;
            lastDelta = true;
            continue;
        }

        const Onb onb = makeOnb(h.N);
        const float3 wo = toLocal(onb, rd * -1.0f);
        const BsdfLobes lobes = makeLobes(s, fmaxf(wo.z, 1e-4f));

        // ---- éclairage direct : échantillonnage des luminaires + rayon d'ombre + MIS
        if (params.numLights > 0) {
            const unsigned li = min(static_cast<unsigned>(rng.next() * params.numLights), params.numLights - 1);
            const RectLight& L = params.lights[li];
            const float2 u = rng.next2();
            const float3 lp = L.corner + L.edgeU * u.x + L.edgeV * u.y;
            float3 d = lp - h.P;
            const float dist2 = dot(d, d);
            const float dist = sqrtf(dist2);
            d = d * (1.0f / dist);
            const float cosL = -dot(d, L.normal);
            const float nol = dot(h.N, d);
            if (cosL > 0.0f && nol > 0.0f && dot(h.Ng, d) > 0.0f) {
                float pdfB;
                const float3 f = evalBsdf(lobes, wo, toLocal(onb, d), pdfB);
                if (maxComp(f) > 0.0f) {
                    const float pdfL = dist2 / (cosL * L.area * params.numLights);
                    const float3 o = offsetRay(h.P, h.Ng, d);
                    if (!occluded(o, d, dist * 0.999f)) {
                        const float w = pdfL * pdfL / (pdfL * pdfL + pdfB * pdfB);
                        float3 c = throughput * f * L.emission * (nol * w / pdfL);
                        if (clampFire && bounce > 0) c = clampLum(c, 30.0f);
                        out.radiance += c;
                    }
                }
            }
        }

        // ---- rebond indirect
        float3 wiL, f;
        float pdf;
        if (!sampleBsdf(lobes, wo, rng, wiL, f, pdf)) break;
        const float3 wi = toWorld(onb, wiL);
        if (dot(wi, h.Ng) <= 0.0f) break;
        throughput *= f * (wiL.z / pdf);
        lastPdf = pdf;
        lastDelta = false;
        ro = offsetRay(h.P, h.Ng, wi);
        rd = wi;

        // roulette russe
        if (bounce >= 2) {
            const float q = fmaxf(0.05f, 1.0f - maxComp(throughput));
            if (rng.next() < q) break;
            throughput = throughput * (1.0f / (1.0f - q));
        }
    }
    return out;
}

}  // namespace

// ============================================================================
extern "C" __global__ void __raygen__render()
{
    const uint3 idx = optixGetLaunchIndex();
    const unsigned pixel = idx.y * params.size.x + idx.x;
    Rng rng{pcgHash(pixel * 9781u + pcgHash(params.frameIndex * 6271u + 17u))};

    const bool accumulate = (params.flags & kFlagAccumulate) != 0;
    const bool dlss = (params.flags & kFlagDLSS) != 0;

    float3 radiance = make_float3(0, 0, 0);
    PathOutput first{};
    const unsigned spp = max(1u, params.spp);
    for (unsigned s = 0; s < spp; ++s) {
        float2 pix;
        if (dlss) {
            // Tous les échantillons au point décalé par le jitter DLSS (le contenu est
            // décalé de +jitter : l'échantillon se trouve donc à centre - jitter).
            pix = make_float2(idx.x + 0.5f - params.jitter.x, idx.y + 0.5f - params.jitter.y);
        } else {
            const float2 j = rng.next2();
            pix = make_float2(idx.x + j.x, idx.y + j.y);
        }
        const PathOutput po = tracePath(pix, rng, s == 0);
        radiance += po.radiance;
        if (s == 0) first = po;
    }
    radiance = radiance * (1.0f / spp);
    if (!(radiance.x == radiance.x) || !(radiance.y == radiance.y) || !(radiance.z == radiance.z))
        radiance = make_float3(0, 0, 0);  // garde-fou NaN

    float4 color = make_float4v(radiance, 1.0f);
    float4 albedo = make_float4v(first.albedo, 1.0f);
    float4 normal = make_float4v(first.normal, 0.0f);
    if (accumulate && params.accumCount > 0) {
        const float n = static_cast<float>(params.accumCount);
        const float inv = 1.0f / (n + 1.0f);
        const float4 a = params.accum[pixel];
        color = make_float4((a.x * n + color.x) * inv, (a.y * n + color.y) * inv, (a.z * n + color.z) * inv, 1.0f);
        const float4 pa = params.albedo[pixel], pn = params.normal[pixel];
        albedo = make_float4((pa.x * n + albedo.x) * inv, (pa.y * n + albedo.y) * inv, (pa.z * n + albedo.z) * inv, 1.0f);
        normal = make_float4((pn.x * n + normal.x) * inv, (pn.y * n + normal.y) * inv, (pn.z * n + normal.z) * inv, 0.0f);
    }
    if (accumulate) params.accum[pixel] = color;
    params.color[pixel] = color;
    params.albedo[pixel] = albedo;
    params.normal[pixel] = normal;
    params.depth[pixel] = first.depth;
    params.motion[pixel] = first.motion;
}

// Sélection d'une touche : un rayon unique lancé depuis la caméra sous le curseur.
extern "C" __global__ void __raygen__pick()
{
    const float3 d = cameraRay(params.cam, params.pickPixel, params.size);
    optixTraverse(params.handle, params.cam.eye, d, 0.0f, 1e16f, 0.0f, kMaskSolid, OPTIX_RAY_FLAG_NONE, 0, 1, 0);
    int id = -1;
    if (optixHitObjectIsHit()) id = params.instances[optixHitObjectGetInstanceIndex()].pickId;
    params.pickResult[0] = id;
}

extern "C" __global__ void __miss__noop() {}

extern "C" __global__ void __closesthit__noop() {}

// Any-hit de la grille perforée : invoqué uniquement pour les micro-triangles OMM
// classés "indéterminés" (les zones pleines/vides sont résolues par le matériel).
extern "C" __global__ void __anyhit__cutout()
{
    const InstanceData& inst = params.instances[optixGetInstanceIndex()];
    const GeometryData& g = params.geometries[inst.geometry];
    if (!g.uvs) return;
    const uint3 tri = g.indices[optixGetPrimitiveIndex()];
    const float2 bc = optixGetTriangleBarycentrics();
    const float2 uv = g.uvs[tri.x] * (1.0f - bc.x - bc.y) + g.uvs[tri.y] * bc.x + g.uvs[tri.z] * bc.y;
    if (grille::sdf(uv) < 0.0f) optixIgnoreIntersection();
}
