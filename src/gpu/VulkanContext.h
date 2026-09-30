// CalculatoRTX - contexte Vulkan commun aux deux backends.
//
//   * backend NVIDIA : Vulkan sert à la présentation (swapchain) et au DLSS (NGX) ; les
//     images viennent de CUDA/OptiX par mémoire externe partagée + sémaphore "timeline"
//     exporté (interopérabilité CUDA <-> Vulkan, compilée seulement avec CUDA) ;
//   * backend Vulkan (AMD, Intel, NVIDIA) : ray tracing matériel par VK_KHR_ray_query,
//     calcul FP64 en compute shader, présentation.
// Le contexte peut aussi être créé sans fenêtre (tests automatiques).
#pragma once

#include "VulkanCommon.h"

#if defined(CRTX_WITH_CUDA)
#include <cuda_runtime.h>
#endif

#include <string>
#include <vector>

namespace crtx {

#if defined(CRTX_WITH_CUDA)
// Tampon Vulkan dont la mémoire est aussi visible par CUDA (zéro copie).
struct SharedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    cudaExternalMemory_t cudaMem = nullptr;
    void* cudaPtr = nullptr;
};
#endif

// Tampon Vulkan ordinaire (éventuellement projeté en mémoire hôte).
struct Buffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    VkDeviceAddress address = 0;  // si créé avec VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT
    void* mapped = nullptr;       // si mémoire HOST_VISIBLE
};

struct GpuImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
};

// Capacités demandées au périphérique logique.
struct DeviceRequest {
    bool cudaInterop = false;  // mémoire / sémaphores exportables (backend NVIDIA)
    bool rayTracing = false;   // VK_KHR_acceleration_structure + VK_KHR_ray_query
    bool float64 = false;      // shaderFloat64 + shaderInt64 (moteur de calcul Vulkan)
    std::vector<std::string> extraExtensions;  // ex. extensions exigées par NGX/DLSS
};

// Fonctions des extensions de ray tracing (chargées par vkGetDeviceProcAddr).
struct RayTracingFns {
    PFN_vkCreateAccelerationStructureKHR createAS = nullptr;
    PFN_vkDestroyAccelerationStructureKHR destroyAS = nullptr;
    PFN_vkGetAccelerationStructureBuildSizesKHR buildSizes = nullptr;
    PFN_vkCmdBuildAccelerationStructuresKHR cmdBuild = nullptr;
    PFN_vkGetAccelerationStructureDeviceAddressKHR asAddress = nullptr;
    PFN_vkCmdWriteAccelerationStructuresPropertiesKHR cmdWriteProps = nullptr;
    PFN_vkCmdCopyAccelerationStructureKHR cmdCopy = nullptr;
};

class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // headless : pas de surface ni de swapchain (tests)
    void createInstance(const std::vector<std::string>& extraInstanceExtensions, bool validation, bool headless = false);
    // Backend NVIDIA : même GPU que le périphérique CUDA (UUID).
    void pickPhysicalDeviceByUuid(const unsigned char uuid[16]);
    // Backend Vulkan : meilleur GPU compatible (dédié de préférence), ou l'index demandé (>= 0).
    void pickPhysicalDevice(const DeviceRequest& req, int preferredIndex);
    // Vérifie qu'un GPU offre les capacités demandées ; 'why' reçoit la raison sinon.
    static bool deviceSupports(VkPhysicalDevice d, const DeviceRequest& req, std::string* why);
    void createDevice(GLFWwindow* window, const DeviceRequest& req);
    void createSwapchain(uint32_t width, uint32_t height, bool vsync);
    void destroySwapchain();

#if defined(CRTX_WITH_CUDA)
    SharedBuffer createSharedBuffer(VkDeviceSize size);
    void destroySharedBuffer(SharedBuffer& b);
    cudaExternalSemaphore_t cudaTimeline() const { return cudaTimeline_; }
#endif
    Buffer createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props);
    void destroyBuffer(Buffer& b);
    GpuImage createImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage);
    void destroyImage(GpuImage& img);

    // Exécute immédiatement des commandes et attend leur fin.
    template <typename Fn>
    void immediateSubmit(Fn&& record)
    {
        VkCommandBuffer cmd = beginOneTime();
        record(cmd);
        endOneTime(cmd);
    }

    static void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                             VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
                             VkAccessFlags dstAccess);
    // Barrière mémoire globale (tampons) entre deux étapes
    static void memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                              VkPipelineStageFlags dstStage, VkAccessFlags dstAccess);

    void waitTimeline(uint64_t value);
    uint64_t timelineValue() const;

    // ---- présentation d'un tampon (1 uint32 par pixel, taille de la swapchain)
    VkResult acquireImage(uint32_t& index);  // VK_SUCCESS, VK_SUBOPTIMAL_KHR ou VK_ERROR_OUT_OF_DATE_KHR
    void recordCopyToSwapchain(VkCommandBuffer cmd, VkBuffer src, uint32_t index) const;
    // Soumet 'cmd' (attend imageAvailable + timeline >= waitValue si waitValue > 0), signale
    // renderFinished(index) et la timeline à signalValue, puis présente. false si la
    // swapchain doit être recréée.
    bool submitAndPresent(VkCommandBuffer cmd, uint32_t index, uint64_t waitValue, uint64_t signalValue);
    // Avance la timeline sans rien présenter (image perdue lors d'un redimensionnement)
    void signalTimeline(uint64_t waitValue, uint64_t signalValue);

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physical_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queueFamily() const { return queueFamily_; }
    // File "compute" asynchrone (famille sans graphique, ex. AMD) ou, à défaut, la file principale.
    VkQueue computeQueue() const { return computeQueue_ ? computeQueue_ : queue_; }
    uint32_t computeFamily() const { return computeQueue_ ? computeFamily_ : queueFamily_; }
    uint32_t timestampBits(uint32_t family) const { return family < timestampBits_.size() ? timestampBits_[family] : 0; }
    VkCommandPool commandPool() const { return commandPool_; }
    VkSwapchainKHR swapchain() const { return swapchain_; }
    VkExtent2D swapExtent() const { return swapExtent_; }
    const std::vector<VkImage>& swapImages() const { return swapImages_; }
    bool swapIsBgra() const { return swapBgra_; }
    VkSemaphore timeline() const { return timeline_; }
    VkSemaphore imageAvailable() const { return imageAvailable_; }
    VkSemaphore renderFinished(uint32_t i) const { return renderFinished_[i]; }
    const std::string& deviceName() const { return deviceName_; }
    uint32_t vendorId() const { return vendorId_; }
    bool isDiscrete() const { return discrete_; }
    const VkPhysicalDeviceLimits& limits() const { return limits_; }
    float timestampPeriodNs() const { return limits_.timestampPeriod; }
    const RayTracingFns& rt() const { return rt_; }
    uint32_t asScratchAlignment() const { return asScratchAlignment_; }

private:
    VkCommandBuffer beginOneTime();
    void endOneTime(VkCommandBuffer cmd);
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    void createTimelineSemaphore(bool exportable);
    void setPhysical(VkPhysicalDevice d);

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkQueue computeQueue_ = VK_NULL_HANDLE;
    uint32_t computeFamily_ = 0;
    std::vector<uint32_t> timestampBits_;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;
    bool headless_ = false;
    bool bufferDeviceAddress_ = false;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkExtent2D swapExtent_{};
    VkFormat swapFormat_ = VK_FORMAT_UNDEFINED;
    bool swapBgra_ = true;
    std::vector<VkImage> swapImages_;
    std::vector<VkSemaphore> renderFinished_;
    VkSemaphore imageAvailable_ = VK_NULL_HANDLE;

    VkSemaphore timeline_ = VK_NULL_HANDLE;
#if defined(CRTX_WITH_CUDA)
    cudaExternalSemaphore_t cudaTimeline_ = nullptr;
#endif

    std::string deviceName_;
    uint32_t vendorId_ = 0;
    bool discrete_ = false;
    VkPhysicalDeviceLimits limits_{};
    RayTracingFns rt_;
    uint32_t asScratchAlignment_ = 256;
};

}  // namespace crtx
