// CalculatoRTX - contexte Vulkan : présentation (swapchain), images DLSS et interopérabilité
// CUDA <-> Vulkan (mémoire externe partagée + sémaphore "timeline" exporté).
//
// Vulkan ne dessine rien : il ne sert qu'à présenter l'image calculée par CUDA/OptiX et à
// fournir au DLSS (NGX) les ressources Vulkan qu'il exige.
#pragma once

#include "VulkanCommon.h"

#include <cuda_runtime.h>

#include <string>
#include <vector>

namespace crtx {

// Tampon Vulkan dont la mémoire est aussi visible par CUDA (zéro copie).
struct SharedBuffer {
    VkBuffer buffer = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkDeviceSize size = 0;
    cudaExternalMemory_t cudaMem = nullptr;
    void* cudaPtr = nullptr;
};

struct GpuImage {
    VkImage image = VK_NULL_HANDLE;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkImageView view = VK_NULL_HANDLE;
    VkFormat format = VK_FORMAT_UNDEFINED;
    uint32_t width = 0, height = 0;
};

class VulkanContext {
public:
    VulkanContext() = default;
    ~VulkanContext();
    VulkanContext(const VulkanContext&) = delete;
    VulkanContext& operator=(const VulkanContext&) = delete;

    // cudaUuid : UUID du GPU CUDA, pour choisir le même VkPhysicalDevice.
    void createInstance(const std::vector<std::string>& extraInstanceExtensions, bool validation);
    void pickPhysicalDevice(const unsigned char cudaUuid[16]);
    void createDevice(GLFWwindow* window, const std::vector<std::string>& extraDeviceExtensions);
    void createSwapchain(uint32_t width, uint32_t height, bool vsync);
    void destroySwapchain();

    SharedBuffer createSharedBuffer(VkDeviceSize size);
    void destroySharedBuffer(SharedBuffer& b);
    GpuImage createImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage);
    void destroyImage(GpuImage& img);

    // Exécute immédiatement des commandes (création de la fonctionnalité DLSS, etc.)
    template <typename Fn>
    void immediateSubmit(Fn&& record)
    {
        VkCommandBuffer cmd = beginOneTime();
        record(cmd);
        endOneTime(cmd);
    }

    // Transitions de layout d'image
    static void imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                             VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
                             VkAccessFlags dstAccess);

    void waitTimeline(uint64_t value);
    uint64_t timelineValue() const;

    VkInstance instance() const { return instance_; }
    VkPhysicalDevice physicalDevice() const { return physical_; }
    VkDevice device() const { return device_; }
    VkQueue queue() const { return queue_; }
    uint32_t queueFamily() const { return queueFamily_; }
    VkCommandPool commandPool() const { return commandPool_; }
    VkSwapchainKHR swapchain() const { return swapchain_; }
    VkExtent2D swapExtent() const { return swapExtent_; }
    const std::vector<VkImage>& swapImages() const { return swapImages_; }
    bool swapIsBgra() const { return swapBgra_; }
    VkSemaphore timeline() const { return timeline_; }
    cudaExternalSemaphore_t cudaTimeline() const { return cudaTimeline_; }
    VkSemaphore imageAvailable() const { return imageAvailable_; }
    VkSemaphore renderFinished(uint32_t i) const { return renderFinished_[i]; }
    const std::string& deviceName() const { return deviceName_; }

private:
    VkCommandBuffer beginOneTime();
    void endOneTime(VkCommandBuffer cmd);
    uint32_t findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const;
    void createTimelineSemaphore();

    VkInstance instance_ = VK_NULL_HANDLE;
    VkDebugUtilsMessengerEXT messenger_ = VK_NULL_HANDLE;
    VkPhysicalDevice physical_ = VK_NULL_HANDLE;
    VkDevice device_ = VK_NULL_HANDLE;
    VkQueue queue_ = VK_NULL_HANDLE;
    uint32_t queueFamily_ = 0;
    VkSurfaceKHR surface_ = VK_NULL_HANDLE;
    VkCommandPool commandPool_ = VK_NULL_HANDLE;

    VkSwapchainKHR swapchain_ = VK_NULL_HANDLE;
    VkExtent2D swapExtent_{};
    VkFormat swapFormat_ = VK_FORMAT_UNDEFINED;
    bool swapBgra_ = true;
    std::vector<VkImage> swapImages_;
    std::vector<VkSemaphore> renderFinished_;
    VkSemaphore imageAvailable_ = VK_NULL_HANDLE;

    VkSemaphore timeline_ = VK_NULL_HANDLE;
    cudaExternalSemaphore_t cudaTimeline_ = nullptr;

    std::string deviceName_;
};

}  // namespace crtx
