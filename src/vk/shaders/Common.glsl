// CalculatoRTX - déclarations communes des shaders du backend Vulkan.
//
// Les structures reprennent octet pour octet src/render/SceneTypes.h (disposition "scalar",
// VK_EXT_scalar_block_layout) ; les tampons sont atteints par leur adresse GPU
// (VK_KHR_buffer_device_address) : aucune copie, aucun descripteur par maillage.
#ifndef CRTX_COMMON_GLSL
#define CRTX_COMMON_GLSL

const float kPi = 3.14159265358979323846;
const float kTwoPi = 6.28318530717958647692;
const float kInvPi = 0.31830988618379067154;

const uint kMaskSolid = 0x01u;
const uint kMaskGlass = 0x02u;
const uint kMaskLight = 0x04u;
const uint kMaskAll = 0xFFu;

const int kPatternNone = 0, kPatternWood = 1, kPatternBrushed = 2, kPatternSolar = 3, kPatternGrille = 4,
          kPatternVfd = 5, kPatternCarbon = 6, kPatternPcb = 7;

const uint kFlagAccumulate = 1u << 1;
const uint kFlagJitter = 1u << 2;
const uint kFlagFireflyClamp = 1u << 3;

struct Material {
    vec3 baseColor;
    float roughness;
    vec3 emission;
    float metallic;
    float transmission;
    float ior;
    float clearcoat;
    float clearcoatRoughness;
    float specular;
    int pattern;
    float patternScale;
    float absorbDistance;
    vec3 absorbColor;
    float haze;
    vec3 hazeColor;
    float emitUpOnly;
};

struct InstanceData {
    float prevObjectToWorld[12];
    int geometry;
    int material;
    int pickId;
    int lightIndex;
    float glow;
    vec3 glowColor;
};

struct RectLight {
    vec3 corner;
    vec3 edgeU;
    vec3 edgeV;
    vec3 normal;
    vec3 emission;
    float area;
};

struct CameraData {
    vec3 eye;
    vec3 U;
    vec3 V;
    vec3 W;
    float zNear;
    float zFar;
};

// Adresses GPU des attributs d'un maillage (0 = absent)
struct GeometryData {
    uint64_t positions;
    uint64_t normals;
    uint64_t uvs;
    uint64_t indices;
    uint alphaCutout;
    uint pad0;
};

layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Vec3Buf { vec3 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer Vec2Buf { vec2 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer UVec3Buf { uvec3 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer MaterialBuf { Material m[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer InstanceBuf { InstanceData d[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) readonly buffer GeometryBuf { GeometryData g[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) readonly buffer LightBuf { RectLight l[]; };
layout(buffer_reference, scalar, buffer_reference_align = 16) buffer Vec4Buf { vec4 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer FloatBuf { float v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 8) buffer Vec2RWBuf { vec2 v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer IntBuf { int v[]; };
layout(buffer_reference, scalar, buffer_reference_align = 4) buffer UintBuf { uint v[]; };

// Paramètres d'une image (miroir de VkFrameParams dans src/vk/VkRenderer.h)
struct FrameParams {
    uint64_t color;       // Vec4Buf  radiance (moyenne accumulée)
    uint64_t albedo;      // Vec4Buf
    uint64_t normal;      // Vec4Buf  (repère caméra)
    uint64_t depth;       // FloatBuf profondeur linéaire le long de W
    uint64_t motion;      // Vec2RWBuf vecteurs de mouvement (pixels, vers l'image précédente)
    uint64_t materials;   // MaterialBuf
    uint64_t geometries;  // GeometryBuf
    uint64_t instances;   // InstanceBuf
    uint64_t lights;      // LightBuf
    uint64_t pickResult;  // IntBuf
    CameraData cam;
    CameraData prevCam;
    uint width;
    uint height;
    uint frameIndex;
    uint accumCount;
    uint spp;
    uint maxBounces;
    uint flags;
    uint numLights;
    float jitterX;
    float jitterY;
    float envIntensity;
    float pickX;
    float pickY;
    float pad0;
};

float saturate(float v) { return clamp(v, 0.0, 1.0); }
float luminance(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }
float maxComp(vec3 a) { return max(a.x, max(a.y, a.z)); }

// point' = M * (p, 1) pour une matrice 3x4 en lignes (format OptiX / SceneTypes.h)
vec3 xformPoint12(float m[12], vec3 p)
{
    return vec3(m[0] * p.x + m[1] * p.y + m[2] * p.z + m[3], m[4] * p.x + m[5] * p.y + m[6] * p.z + m[7],
                m[8] * p.x + m[9] * p.y + m[10] * p.z + m[11]);
}

#endif
