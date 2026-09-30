// CalculatoRTX - moteur de calcul GPU du backend Vulkan (compute shader FP64).
//
// Même algorithme que la version CUDA (analyse, 4 arithmétiques, formatage), exécuté sur
// la file "compute" asynchrone quand le GPU en possède une (AMD) : le calcul ne fait pas
// la queue derrière l'image en cours de rendu.
#pragma once

#include "../calc/ICalcEngine.h"
#include "VkPipeline.h"

namespace crtx {

class VkCalcEngine : public calc::ICalcEngine {
public:
    explicit VkCalcEngine(VulkanContext& vk);
    ~VkCalcEngine() override;
    VkCalcEngine(const VkCalcEngine&) = delete;
    VkCalcEngine& operator=(const VkCalcEngine&) = delete;

    calc::Result evaluate(const std::string& program, calc::AngleMode mode, double ansHi, double ansLo, double memHi,
                          double memLo, float* gpuMicroseconds = nullptr) override;
    const char* apiName() const override { return "VK"; }

private:
    VulkanContext& vk_;
    ComputePipeline pipe_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    VkDescriptorSet set_ = VK_NULL_HANDLE;
    VkCommandPool cmdPool_ = VK_NULL_HANDLE;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    VkFence fence_ = VK_NULL_HANDLE;
    VkQueryPool queries_ = VK_NULL_HANDLE;
    Buffer request_, result_;
};

}  // namespace crtx
