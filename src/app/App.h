// CalculatoRTX - application : fenêtre, boucle de rendu, synchronisation CUDA <-> Vulkan.
#pragma once

#include "../calc/CalcEngine.h"
#include "../dlss/DlssUpscaler.h"
#include "../gpu/VulkanContext.h"
#include "../render/Kernels.h"
#include "../render/OptixRenderer.h"
#include "../scene/CalculatorScene.h"
#include "../scene/StrokeFont.h"
#include "CalculatorController.h"

#include <memory>
#include <string>

namespace crtx {

struct AppOptions {
    int width = 1600;
    int height = 1000;
    bool validation = false;
    bool dlss = true;
    DlssMode dlssMode = DlssMode::Quality;
    bool vsync = true;
    int cudaDevice = 0;
};

class App {
public:
    explicit App(const AppOptions& opt);
    ~App();
    int run();

private:
    struct Orbit {
        float yaw = 0.0f;
        float pitch = 0.99f;  // ~57°
        float distance = 29.0f;
        float fovY = 0.54f;   // ~31°
    };

    void initCuda();
    void initWindow();
    void initVulkanAndDlss();
    void configureResolution();
    void releaseFrameResources();
    void frame(float dt);
    CameraData makeCamera(const Orbit& o, uint32_t w, uint32_t h) const;
    void onKeyChar(unsigned int codepoint);
    void onKey(int key, int action, int mods);
    void onMouseButton(int button, int action);
    void onCursor(double x, double y);
    void onScroll(double dy);
    void pressKey(KeyId id);
    void printHelp() const;
    void updateTitle(float dt);
    void cudaSignal(uint64_t value);
    void cudaWait(uint64_t value);

    AppOptions opt_;
    GLFWwindow* window_ = nullptr;
    cudaStream_t stream_ = nullptr;
    unsigned char cudaUuid_[16] = {};
    std::string gpuName_;

    std::unique_ptr<VulkanContext> vk_;
    std::unique_ptr<DlssUpscaler> dlss_;
    std::unique_ptr<calc::CalcEngine> engine_;
    std::unique_ptr<CalculatorController> controller_;
    StrokeFont font_;
    std::unique_ptr<CalculatorScene> scene_;
    std::unique_ptr<OptixRenderer> renderer_;
    std::unique_ptr<PostProcessor> post_;

    // ressources dépendant de la résolution
    uint32_t displayW_ = 0, displayH_ = 0, renderW_ = 0, renderH_ = 0;
    bool dlssActive_ = false;
    SharedBuffer shColor_, shDepth_, shMotion_, shDlssOut_, shPresent_;
    GpuImage imgColor_, imgDepth_, imgMotion_, imgOut_;
    VkCommandBuffer cmdDlss_ = VK_NULL_HANDLE, cmdPresent_ = VK_NULL_HANDLE;
    bool swapchainDirty_ = false;
    bool resetHistory_ = true;
    unsigned jitterPhases_ = 8;
    float jitterSign_ = 1.0f;

    // état de rendu
    RenderSettings settings_;
    PostProcessor::Params postParams_;
    uint64_t frameCount_ = 0;
    Orbit orbit_, orbitTarget_;
    CameraData prevCamera_{};
    bool hasPrevCamera_ = false;

    // entrées
    double cursorX_ = 0, cursorY_ = 0;
    bool rotating_ = false;
    double dragX_ = 0, dragY_ = 0;
    int hoveredKey_ = -1;

    // statistiques
    float titleTimer_ = 0.0f;
    float fpsAccum_ = 0.0f;
    int fpsFrames_ = 0;
    float lastFrameMs_ = 16.0f;
};

}  // namespace crtx
