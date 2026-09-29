// CalculatoRTX - paramètres partagés entre l'hôte et les programmes OptiX (GPU).
#pragma once

#include <cuda_runtime.h>
#include <optix_types.h>

#include "../common/VecMath.h"

namespace crtx {

// Masques de visibilité des instances (8 bits OptiX)
constexpr unsigned kMaskSolid = 0x01;  // bloque la lumière
constexpr unsigned kMaskGlass = 0x02;  // verre : ignoré par les rayons d'ombre
constexpr unsigned kMaskLight = 0x04;  // luminaires : visibles, mais ne font pas d'ombre
constexpr unsigned kMaskAll = 0xFF;

// Motifs procéduraux évalués dans le shader (aucune texture)
enum Pattern : int {
    kPatternNone = 0,
    kPatternWood = 1,      // bureau en noyer verni
    kPatternBrushed = 2,   // aluminium brossé
    kPatternSolar = 3,     // cellules photovoltaïques
    kPatternGrille = 4,    // grille perforée (trous hexagonaux, OMM + any-hit)
    kPatternVfd = 5,       // fond d'afficheur fluorescent (trame fine)
    kPatternCarbon = 6,    // fibre de carbone (lit des touches)
    kPatternPcb = 7,       // circuit imprimé : vernis épargne, pistes, vias
};

struct Material {
    float3 baseColor;
    float roughness;
    float3 emission;
    float metallic;
    float transmission;  // 1 = diélectrique transparent (verre, coque translucide)
    float ior;
    float clearcoat;
    float clearcoatRoughness;
    float specular;  // réflectance spéculaire des diélectriques (0.5 => F0 = 4 %)
    int pattern;
    float patternScale;
    // Absorption volumique (loi de Beer-Lambert) des diélectriques : la lumière qui a
    // parcouru absorbDistance dans la matière est teintée par absorbColor (0 = aucune).
    float absorbDistance;
    float3 absorbColor;
    // Plastique translucide : fraction 'haze' de la lumière qui entre dans la matière est
    // rediffusée vers l'extérieur avec l'albédo hazeColor (diffusion volumique approchée).
    float haze;
    float3 hazeColor;
    float emitUpOnly;  // 1 = n'émet que vers le haut (+Y objet) : impression lumineuse sur plastique transparent
};

struct GeometryData {
    const float3* positions;
    const float3* normals;  // nullptr => ombrage plat
    const float2* uvs;      // peut être nullptr
    const uint3* indices;
};

struct InstanceData {
    float prevObjectToWorld[12];  // transformation de l'image précédente (vecteurs de mouvement)
    int geometry;
    int material;
    int pickId;      // -1 = non cliquable, sinon indice de touche
    int lightIndex;  // -1 = pas un luminaire échantillonné explicitement
    float glow;      // surbrillance (survol / appui)
    float3 glowColor;
};

struct RectLight {
    float3 corner;
    float3 edgeU;
    float3 edgeV;
    float3 normal;
    float3 emission;
    float area;
};

struct CameraData {
    float3 eye;
    float3 U;  // droite * tan(fov/2) * aspect
    float3 V;  // haut * tan(fov/2)
    float3 W;  // direction de visée (unitaire)
    float zNear;
    float zFar;
};

enum LaunchFlags : unsigned {
    kFlagSER = 1u << 0,          // Shader Execution Reordering (Ada)
    kFlagAccumulate = 1u << 1,   // accumulation progressive (mode sans DLSS)
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
