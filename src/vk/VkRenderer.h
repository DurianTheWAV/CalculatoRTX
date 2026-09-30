// CalculatoRTX - moteur de rendu du backend Vulkan (AMD Radeon RX 6000+, Intel Arc, NVIDIA RTX).
//
//   * structures d'accélération matérielles : un BLAS par maillage, un TLAS reconstruit à
//     chaque image (animation des touches, inclinaison) ;
//   * path tracing par VK_KHR_ray_query dans un compute shader (PathTrace.comp) ;
//   * débruiteur spatial à-trous guidé (albédo, normales, profondeur) ;
//   * bloom, tone mapping ACES, puis AMD FidelityFX Super Resolution 1 (EASU + RCAS) quand
//     la résolution de rendu est inférieure à celle de l'affichage ;
//   * sélection de la touche sous le curseur par un rayon (Pick).
// Toutes les données sont lues par adresse GPU (buffer device address).
#pragma once

#include "../scene/CalculatorScene.h"
#include "VkPipeline.h"

#include <cstdint>
#include <vector>

namespace crtx {

// Miroir exact de FrameParams (src/vk/shaders/Common.glsl), disposition scalar.
struct VkFrameParams {
    uint64_t color, albedo, normal, depth, motion, materials, geometries, instances, lights, pickResult;
    CameraData cam;
    CameraData prevCam;
    uint32_t width, height, frameIndex, accumCount, spp, maxBounces, flags, numLights;
    float jitterX, jitterY, envIntensity, pickX, pickY, pad0;
};
static_assert(sizeof(VkFrameParams) == 248, "VkFrameParams : disposition partagée avec les shaders");

struct VkFrameDesc {
    CameraData camera;
    CameraData prevCamera;
    uint32_t renderW = 0, renderH = 0;  // zone rendue (<= affichage) ; FSR 1 si plus petite
    bool runPost = true;                // false : accumule sans produire d'image (elle reste affichée)
    bool temporal = false;              // image en mouvement : accumulation temporelle reprojetée
    uint32_t frameIndex = 0;
    bool resetAccumulation = true;
    uint32_t spp = 1;
    uint32_t maxBounces = 5;
    bool fireflyClamp = true;
    bool denoise = true;
    float pickX = 0.0f, pickY = 0.0f;  // pixel de rendu sous le curseur
    // post-traitement
    float exposure = 1.0f;
    float bloomStrength = 0.08f;  // part de l'énergie au-dessus du seuil étalée en halo
    float bloomThreshold = 1.2f;
    float vignette = 0.22f;
    float sharpness = 0.25f;  // RCAS : 0 = maximum, en "stops" d'atténuation
    bool bgra = true;
};

class VkRenderer {
public:
    explicit VkRenderer(VulkanContext& vk);
    ~VkRenderer();
    VkRenderer(const VkRenderer&) = delete;
    VkRenderer& operator=(const VkRenderer&) = delete;

    // À appeler une fois l'image précédente terminée : libère les ressources différées.
    void beginFrame();
    // Taille d'affichage (swapchain) ; le rendu se fait sur tout ou partie de cette surface.
    void resize(uint32_t displayW, uint32_t displayH);
    // Met à jour la scène (BLAS des maillages modifiés, TLAS, matériaux) dans 'cmd'.
    void syncScene(CalculatorScene& scene, VkCommandBuffer cmd);
    // Enregistre une image complète : rendu -> débruitage -> post -> FSR -> octets de la swapchain.
    void recordFrame(VkCommandBuffer cmd, const VkFrameDesc& f);
    // Seulement le rayon de sélection (image convergée, plus de path tracing)
    void recordPick(VkCommandBuffer cmd, const VkFrameDesc& f);

    const Buffer& presentBuffer() const { return present_; }  // uint32 BGRA/RGBA, taille d'affichage
    int pickResult() const;                                    // touche sous le curseur (image précédente)
    uint32_t accumulated() const { return accumCount_; }
    float lastTraceMs() const { return traceMs_; }
    float lastPostMs() const { return postMs_; }
    uint32_t renderWidth() const { return renderW_; }
    uint32_t renderHeight() const { return renderH_; }
    uint32_t displayWidth() const { return displayW_; }
    uint32_t displayHeight() const { return displayH_; }
    void resetAccumulation() { accumCount_ = 0; }

private:
    struct MeshGpu {
        Buffer positions, normals, uvs, indices;
        Buffer asBuffer;
        VkAccelerationStructureKHR blas = VK_NULL_HANDLE;
        VkDeviceAddress blasAddress = 0;
        bool alpha = false;
        uint32_t triangles = 0;
    };
    struct PostPush {
        uint64_t src, dst, aux0, aux1;
        uint32_t srcW, srcH, dstW, dstH;
        float p0, p1, p2, p3;
        uint32_t u0, u1;
    };
    static_assert(sizeof(PostPush) == 72, "PostPush : constantes de poussée partagées avec Post.glsl");
    struct TemporalPush {
        uint64_t color, normal, motion, histIn, histOut, depthIn, depthOut;
        uint32_t width, height, valid;
        float maxHistory;
    };
    static_assert(sizeof(TemporalPush) == 72, "TemporalPush : constantes de poussée partagées avec Temporal.comp");

    void createEnvironment();
    void createDescriptors();
    void freeMesh(MeshGpu& m);
    void deferDestroy(Buffer& b);
    Buffer uploadBuffer(VkCommandBuffer cmd, const void* data, VkDeviceSize bytes, VkBufferUsageFlags usage);
    void buildMeshes(CalculatorScene& scene, VkCommandBuffer cmd);
    void buildTlas(const CalculatorScene& scene, VkCommandBuffer cmd);
    void ensureHostBuffer(Buffer& b, VkDeviceSize bytes, VkBufferUsageFlags usage);
    void post(VkCommandBuffer cmd, const ComputePipeline& p, const PostPush& push, uint32_t w, uint32_t h, uint32_t local);
    void freeFrameBuffers();
    void writeParams(const VkFrameDesc& f);

    VulkanContext& vk_;
    ComputePipeline trace_, pick_, envBake_, denoise_, bloomDown_, bloomBlur_, composite_, easu_, rcas_, pack_, temporal_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet traceSet_ = VK_NULL_HANDLE, pickSet_ = VK_NULL_HANDLE;
    GpuImage env_;
    VkSampler envSampler_ = VK_NULL_HANDLE;

    std::vector<MeshGpu> meshes_;
    std::vector<Buffer> pendingDestroy_;
    std::vector<VkAccelerationStructureKHR> pendingDestroyAs_;
    Buffer tlasBuffer_, tlasScratch_, instanceBuffer_;
    VkAccelerationStructureKHR tlas_ = VK_NULL_HANDLE;
    VkDeviceSize tlasCapacity_ = 0;
    Buffer materials_, geometries_, instances_, lights_, frameUbo_, pickBuf_;
    std::vector<Affine> prevTransforms_;
    uint32_t numLights_ = 0;
    bool tlasDescriptorDirty_ = true;

    // tampons de résolution
    uint32_t renderW_ = 0, renderH_ = 0, displayW_ = 0, displayH_ = 0, bloomW_ = 0, bloomH_ = 0;
    Buffer color_, albedo_, normal_, denoiseA_, denoiseB_, bloomA_, bloomB_, ldr_, easuOut_, rcasOut_, present_;
    Buffer motion_, hist_[2], histDepth_[2];
    int histIndex_ = 0;
    bool historyValid_ = false;
    uint32_t accumCount_ = 0;
    bool geometryChanged_ = true;

    VkQueryPool queries_ = VK_NULL_HANDLE;
    bool queriesWritten_ = false;
    float traceMs_ = 0.0f, postMs_ = 0.0f;
};

}  // namespace crtx
