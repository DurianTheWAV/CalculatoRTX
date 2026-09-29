// CalculatoRTX - intégration NVIDIA DLSS Super Resolution (NGX / Vulkan).
#include "DlssUpscaler.h"

#if defined(CRTX_WITH_DLSS)
#include <nvsdk_ngx_helpers.h>
#include <nvsdk_ngx_helpers_vk.h>
#include <nvsdk_ngx_vk.h>
#endif

#include <cstdio>

namespace crtx {

const char* dlssModeName(DlssMode m)
{
    switch (m) {
        case DlssMode::Dlaa: return "DLAA";
        case DlssMode::Quality: return "Qualite";
        case DlssMode::Balanced: return "Equilibre";
        case DlssMode::Performance: return "Performance";
        case DlssMode::UltraPerformance: return "Ultra Performance";
    }
    return "?";
}

#if defined(CRTX_WITH_DLSS)

namespace {

// Identifiant de projet (moteur "custom", cf. guide de programmation DLSS §5.2)
constexpr const char* kProjectId = "6c1f3a2e-9b4d-4e8a-a7c5-c41c7a70f071";
constexpr const char* kEngineVersion = "1.0.0";

NVSDK_NGX_PerfQuality_Value toNgx(DlssMode m)
{
    switch (m) {
        case DlssMode::Dlaa: return NVSDK_NGX_PerfQuality_Value_DLAA;
        case DlssMode::Quality: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case DlssMode::Balanced: return NVSDK_NGX_PerfQuality_Value_Balanced;
        case DlssMode::Performance: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
        case DlssMode::UltraPerformance: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
    }
    return NVSDK_NGX_PerfQuality_Value_MaxQuality;
}

void NVSDK_CONV ngxLog(const char* message, NVSDK_NGX_Logging_Level, NVSDK_NGX_Feature)
{
    std::fprintf(stderr, "[NGX] %s", message);
}

NVSDK_NGX_FeatureDiscoveryInfo discoveryInfo(NVSDK_NGX_FeatureCommonInfo* common)
{
    NVSDK_NGX_FeatureDiscoveryInfo info{};
    info.SDKVersion = NVSDK_NGX_Version_API;
    info.FeatureID = NVSDK_NGX_Feature_SuperSampling;
    info.Identifier.IdentifierType = NVSDK_NGX_Application_Identifier_Type_Project_Id;
    info.Identifier.v.ProjectDesc.ProjectId = kProjectId;
    info.Identifier.v.ProjectDesc.EngineType = NVSDK_NGX_ENGINE_TYPE_CUSTOM;
    info.Identifier.v.ProjectDesc.EngineVersion = kEngineVersion;
    info.ApplicationDataPath = L".";
    info.FeatureInfo = common;
    return info;
}

NVSDK_NGX_Resource_VK imageResource(VkImage image, VkImageView view, VkFormat format, uint32_t w, uint32_t h, bool rw)
{
    VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    return NVSDK_NGX_Create_ImageView_Resource_VK(view, image, range, format, w, h, rw);
}

}  // namespace

bool DlssUpscaler::compiledIn() { return true; }

std::vector<std::string> DlssUpscaler::requiredInstanceExtensions()
{
    std::vector<std::string> out;
    NVSDK_NGX_FeatureDiscoveryInfo info = discoveryInfo(nullptr);
    uint32_t count = 0;
    VkExtensionProperties* props = nullptr;
    if (NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureInstanceExtensionRequirements(&info, &count, &props)))
        for (uint32_t i = 0; i < count; ++i) out.emplace_back(props[i].extensionName);
    return out;
}

std::vector<std::string> DlssUpscaler::requiredDeviceExtensions(VkInstance instance, VkPhysicalDevice physical)
{
    std::vector<std::string> out;
    NVSDK_NGX_FeatureDiscoveryInfo info = discoveryInfo(nullptr);
    uint32_t count = 0;
    VkExtensionProperties* props = nullptr;
    if (NVSDK_NGX_SUCCEED(NVSDK_NGX_VULKAN_GetFeatureDeviceExtensionRequirements(instance, physical, &info, &count, &props)))
        for (uint32_t i = 0; i < count; ++i) out.emplace_back(props[i].extensionName);
    return out;
}

DlssUpscaler::~DlssUpscaler() { shutdown(); }

bool DlssUpscaler::init(VkInstance instance, VkPhysicalDevice physical, VkDevice device, const std::wstring& dataPath,
                        const std::wstring& dllSearchPath)
{
    const wchar_t* paths[] = {dllSearchPath.c_str()};
    NVSDK_NGX_FeatureCommonInfo common{};
    common.PathListInfo.Path = paths;
    common.PathListInfo.Length = 1;
    common.LoggingInfo.LoggingCallback = &ngxLog;
    common.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
    common.LoggingInfo.DisableOtherLoggingSinks = false;

    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID(
        kProjectId, NVSDK_NGX_ENGINE_TYPE_CUSTOM, kEngineVersion, dataPath.c_str(), instance, physical, device,
        vkGetInstanceProcAddr, vkGetDeviceProcAddr, &common, NVSDK_NGX_Version_API);
    if (NVSDK_NGX_FAILED(r)) {
        status_ = "NGX indisponible (pilote ?) code " + std::to_string(static_cast<unsigned>(r));
        return false;
    }
    device_ = device;  // NGX initialisé : Shutdown1 sera appelé à la fermeture
    NVSDK_NGX_Parameter* params = nullptr;
    r = NVSDK_NGX_VULKAN_GetCapabilityParameters(&params);
    if (NVSDK_NGX_FAILED(r) || !params) {
        status_ = "NGX : paramètres de capacité indisponibles";
        return false;
    }
    params_ = params;
    int supported = 0, needsDriver = 0;
    NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSampling_NeedsUpdatedDriver, &needsDriver);
    const NVSDK_NGX_Result ra = NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSampling_Available, &supported);
    if (needsDriver) {
        status_ = "DLSS : pilote NVIDIA trop ancien";
        return false;
    }
    if (NVSDK_NGX_FAILED(ra) || !supported) {
        int initResult = 0;
        NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSampling_FeatureInitResult, &initResult);
        status_ = "DLSS non disponible (libnvidia-ngx-dlss / nvngx_dlss absent ? code " + std::to_string(initResult) + ")";
        return false;
    }
    available_ = true;
    status_ = "DLSS disponible";
    return true;
}

void DlssUpscaler::shutdown()
{
    releaseFeature();
    if (params_) {
        NVSDK_NGX_VULKAN_DestroyParameters(static_cast<NVSDK_NGX_Parameter*>(params_));
        params_ = nullptr;
    }
    if (device_) {
        NVSDK_NGX_VULKAN_Shutdown1(device_);
        device_ = VK_NULL_HANDLE;
    }
    available_ = false;
}

bool DlssUpscaler::optimalRenderSize(DlssMode mode, uint32_t displayW, uint32_t displayH, uint32_t& renderW,
                                     uint32_t& renderH)
{
    if (!available_) return false;
    if (mode == DlssMode::Dlaa) {
        renderW = displayW;
        renderH = displayH;
        return true;
    }
    unsigned ow = 0, oh = 0, maxW = 0, maxH = 0, minW = 0, minH = 0;
    float sharpness = 0.0f;
    const NVSDK_NGX_Result r = NGX_DLSS_GET_OPTIMAL_SETTINGS(static_cast<NVSDK_NGX_Parameter*>(params_), displayW,
                                                             displayH, toNgx(mode), &ow, &oh, &maxW, &maxH, &minW,
                                                             &minH, &sharpness);
    if (NVSDK_NGX_FAILED(r) || ow == 0 || oh == 0) return false;
    renderW = ow;
    renderH = oh;
    return true;
}

bool DlssUpscaler::createFeature(VkCommandBuffer cmd, DlssMode mode, uint32_t renderW, uint32_t renderH,
                                 uint32_t displayW, uint32_t displayH)
{
    if (!available_) return false;
    releaseFeature();
    NVSDK_NGX_DLSS_Create_Params cp{};
    cp.Feature.InWidth = renderW;
    cp.Feature.InHeight = renderH;
    cp.Feature.InTargetWidth = displayW;
    cp.Feature.InTargetHeight = displayH;
    cp.Feature.InPerfQualityValue = toNgx(mode);
    // Entrée HDR linéaire, vecteurs de mouvement en résolution de rendu sans jitter,
    // exposition automatique (le tone mapping est fait après le DLSS, en CUDA).
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes |
                              NVSDK_NGX_DLSS_Feature_Flags_AutoExposure;
    cp.InEnableOutputSubrects = false;
    NVSDK_NGX_Handle* handle = nullptr;
    const NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSS_EXT1(device_, cmd, 1, 1, &handle,
                                                           static_cast<NVSDK_NGX_Parameter*>(params_), &cp);
    if (NVSDK_NGX_FAILED(r) || !handle) {
        status_ = "Échec de création de la fonctionnalité DLSS";
        return false;
    }
    feature_ = handle;
    return true;
}

void DlssUpscaler::releaseFeature()
{
    if (feature_) {
        if (device_) vkDeviceWaitIdle(device_);
        NVSDK_NGX_VULKAN_ReleaseFeature(static_cast<NVSDK_NGX_Handle*>(feature_));
        feature_ = nullptr;
    }
}

bool DlssUpscaler::evaluate(VkCommandBuffer cmd, const DlssImages& img, uint32_t renderW, uint32_t renderH,
                            uint32_t displayW, uint32_t displayH, float jitterX, float jitterY, bool reset,
                            float frameTimeMs)
{
    if (!feature_) return false;
    NVSDK_NGX_Resource_VK color = imageResource(img.color, img.colorView, img.colorFormat, renderW, renderH, false);
    NVSDK_NGX_Resource_VK depth = imageResource(img.depth, img.depthView, img.depthFormat, renderW, renderH, false);
    NVSDK_NGX_Resource_VK motion = imageResource(img.motion, img.motionView, img.motionFormat, renderW, renderH, false);
    NVSDK_NGX_Resource_VK output = imageResource(img.output, img.outputView, img.outputFormat, displayW, displayH, true);

    NVSDK_NGX_VK_DLSS_Eval_Params ep{};
    ep.Feature.pInColor = &color;
    ep.Feature.pInOutput = &output;
    ep.Feature.InSharpness = 0.0f;
    ep.pInDepth = &depth;
    ep.pInMotionVectors = &motion;
    ep.InJitterOffsetX = jitterX;
    ep.InJitterOffsetY = jitterY;
    ep.InRenderSubrectDimensions.Width = renderW;
    ep.InRenderSubrectDimensions.Height = renderH;
    ep.InReset = reset ? 1 : 0;
    ep.InMVScaleX = 1.0f;  // vecteurs déjà en pixels de rendu
    ep.InMVScaleY = 1.0f;
    ep.InPreExposure = 1.0f;
    ep.InExposureScale = 1.0f;
    ep.InFrameTimeDeltaInMsec = frameTimeMs;
    const NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSS_EXT(cmd, static_cast<NVSDK_NGX_Handle*>(feature_),
                                                            static_cast<NVSDK_NGX_Parameter*>(params_), &ep);
    return NVSDK_NGX_SUCCEED(r);
}

#else  // ------------------------------------------------------------ sans SDK DLSS

bool DlssUpscaler::compiledIn() { return false; }
std::vector<std::string> DlssUpscaler::requiredInstanceExtensions() { return {}; }
std::vector<std::string> DlssUpscaler::requiredDeviceExtensions(VkInstance, VkPhysicalDevice) { return {}; }
DlssUpscaler::~DlssUpscaler() = default;
bool DlssUpscaler::init(VkInstance, VkPhysicalDevice, VkDevice, const std::wstring&, const std::wstring&)
{
    status_ = "DLSS non compilé (option CRTX_ENABLE_DLSS=OFF)";
    return false;
}
void DlssUpscaler::shutdown() {}
bool DlssUpscaler::optimalRenderSize(DlssMode, uint32_t, uint32_t, uint32_t&, uint32_t&) { return false; }
bool DlssUpscaler::createFeature(VkCommandBuffer, DlssMode, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
void DlssUpscaler::releaseFeature() {}
bool DlssUpscaler::evaluate(VkCommandBuffer, const DlssImages&, uint32_t, uint32_t, uint32_t, uint32_t, float, float,
                            bool, float)
{
    return false;
}

#endif

}  // namespace crtx
