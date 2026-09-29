// CalculatoRTX - super-résolution NVIDIA DLSS (NGX, chemin Vulkan) exécutée sur les Tensor cores.
//
// Compilé avec le SDK DLSS officiel si CRTX_WITH_DLSS est défini (téléchargé automatiquement
// par CMake). Sans SDK ou sans GPU RTX, l'application se replie sur un rendu en résolution
// native accumulé + débruiteur IA OptiX.
#pragma once

#include "../gpu/VulkanCommon.h"

#include <string>
#include <vector>

namespace crtx {

enum class DlssMode { Dlaa, Quality, Balanced, Performance, UltraPerformance };

const char* dlssModeName(DlssMode m);

struct DlssImages {
    VkImage color;  VkImageView colorView;  VkFormat colorFormat;
    VkImage depth;  VkImageView depthView;  VkFormat depthFormat;
    VkImage motion; VkImageView motionView; VkFormat motionFormat;
    VkImage output; VkImageView outputView; VkFormat outputFormat;
};

class DlssUpscaler {
public:
    DlssUpscaler() = default;
    ~DlssUpscaler();

    static bool compiledIn();

    // À appeler AVANT vkCreateInstance / vkCreateDevice : extensions exigées par NGX.
    static std::vector<std::string> requiredInstanceExtensions();
    static std::vector<std::string> requiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physical);

    bool init(VkInstance instance, VkPhysicalDevice physical, VkDevice device, const std::wstring& dataPath,
              const std::wstring& dllSearchPath);
    void shutdown();
    bool available() const { return available_; }
    const std::string& status() const { return status_; }

    // Résolution de rendu optimale pour une résolution d'affichage donnée.
    bool optimalRenderSize(DlssMode mode, uint32_t displayW, uint32_t displayH, uint32_t& renderW, uint32_t& renderH);

    // (Re)crée la fonctionnalité DLSS ; cmd doit être soumis puis attendu par l'appelant.
    bool createFeature(VkCommandBuffer cmd, DlssMode mode, uint32_t renderW, uint32_t renderH, uint32_t displayW,
                       uint32_t displayH);
    void releaseFeature();
    bool hasFeature() const { return feature_ != nullptr; }

    bool evaluate(VkCommandBuffer cmd, const DlssImages& img, uint32_t renderW, uint32_t renderH, uint32_t displayW,
                  uint32_t displayH, float jitterX, float jitterY, bool reset, float frameTimeMs);

private:
    bool available_ = false;
    std::string status_ = "DLSS non initialisé";
    VkDevice device_ = VK_NULL_HANDLE;
    void* params_ = nullptr;   // NVSDK_NGX_Parameter*
    void* feature_ = nullptr;  // NVSDK_NGX_Handle*
};

}  // namespace crtx
