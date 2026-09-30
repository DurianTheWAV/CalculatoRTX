// CalculatoRTX - backend Vulkan (AMD / Intel / NVIDIA) : voir VulkanBackend.h.
#include "VulkanBackend.h"

#include "../app/Screenshot.h"
#include "../calc/CalcCore.cuh"
#include "../gpu/VulkanContext.h"
#include "../vk/VkCalcEngine.h"
#include "../vk/VkRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace crtx {

namespace {

// Repli pour les GPU sans FP64 (Intel Arc) : même cœur de calcul, exécuté sur le CPU.
class CpuCalcEngine final : public calc::ICalcEngine {
public:
    calc::Result evaluate(const std::string& program, calc::AngleMode mode, double ansHi, double ansLo, double memHi,
                          double memLo, float* gpuMicroseconds) override
    {
        const calc::Request rq = calc::makeRequest(program, mode, ansHi, ansLo, memHi, memLo);
        calc::Result r{};
        calc::core::evaluateSequential(rq, r);
        if (gpuMicroseconds) *gpuMicroseconds = 0.0f;
        return r;
    }
    const char* apiName() const override { return "CPU"; }
};

// Facteur de réduction de la résolution de rendu en mouvement (AMD FSR 1)
float fsrScale(UpscaleMode m)
{
    switch (m) {
        case UpscaleMode::Native: return 1.0f;
        case UpscaleMode::Balanced: return 1.7f;
        case UpscaleMode::Performance: return 2.0f;
        case UpscaleMode::UltraPerformance: return 3.0f;
        default: return 1.5f;  // Quality
    }
}

UpscaleMode autoMode(uint32_t w, uint32_t h)
{
    const double mp = static_cast<double>(w) * h / 1.0e6;
    return mp <= 2.2 ? UpscaleMode::Quality : (mp <= 3.8 ? UpscaleMode::Balanced : UpscaleMode::Performance);
}

constexpr unsigned kShowStatic = 4;  // images natives accumulées avant de remplacer l'image FSR

class VulkanBackend final : public Backend {
public:
    VulkanBackend(GLFWwindow* window, const BackendOptions& opt) : window_(window), opt_(opt)
    {
        mode_ = opt.mode;
        upscale_ = opt.upscale;
        vk_ = std::make_unique<VulkanContext>();
        vk_->createInstance({}, opt.validation, window == nullptr);
        DeviceRequest req;
        req.rayTracing = true;
        req.float64 = true;
        try {
            vk_->pickPhysicalDevice(req, opt.device);
        } catch (const std::exception&) {
            // GPU à ray tracing sans FP64 (Intel Arc) : rendu Vulkan, calcul sur le CPU
            req.float64 = false;
            vk_->pickPhysicalDevice(req, opt.device);
            fp64_ = false;
        }
        vk_->createDevice(window, req);
        if (fp64_) {
            calc_ = std::make_unique<VkCalcEngine>(*vk_);
        } else {
            CRTX_LOG("Attention : ce GPU n'a pas de calcul double précision (shaderFloat64) ; les calculs de la "
                     "calculatrice sont exécutés sur le CPU");
            calc_ = std::make_unique<CpuCalcEngine>();
        }
        renderer_ = std::make_unique<VkRenderer>(*vk_);
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = vk_->commandPool();
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(vk_->device(), &ai, &cmd_));
        const char* vendor = vk_->vendorId() == 0x1002 ? "AMD" : vk_->vendorId() == 0x10DE ? "NVIDIA"
                           : vk_->vendorId() == 0x8086 ? "Intel" : "autre";
        CRTX_LOG("Backend Vulkan : %s (%s), ray tracing VK_KHR_ray_query, AMD FSR 1", vk_->deviceName().c_str(), vendor);
    }

    ~VulkanBackend() override
    {
        if (vk_ && vk_->device()) vkDeviceWaitIdle(vk_->device());
        renderer_.reset();
        calc_.reset();
        if (vk_ && cmd_) vkFreeCommandBuffers(vk_->device(), vk_->commandPool(), 1, &cmd_);
        vk_.reset();
    }

    std::string name() const override { return "Vulkan"; }
    std::string deviceName() const override { return vk_->deviceName(); }
    calc::ICalcEngine& calc() override { return *calc_; }
    int hoveredKey() const override { return hovered_; }
    bool idle() const override { return idle_; }
    float aspect() const override
    {
        return static_cast<float>(displayW_) / static_cast<float>(std::max(displayH_, 1u));
    }
    std::string upscalerName() const override { return "FSR 1"; }
    bool upscaleEnabled() const override { return upscale_; }
    UpscaleMode upscaleMode() const override { return mode_; }
    void setVsync(bool vsync) override { opt_.vsync = vsync; }
    void setUpscale(bool enabled, UpscaleMode mode) override
    {
        upscale_ = enabled;
        mode_ = mode;
    }

    void resize(uint32_t fbw, uint32_t fbh) override
    {
        waitIdle();
        if (window_) {
            vk_->createSwapchain(fbw, fbh, opt_.vsync);
            displayW_ = vk_->swapExtent().width;
            displayH_ = vk_->swapExtent().height;
        } else {
            displayW_ = fbw;
            displayH_ = fbh;
        }
        renderer_->resize(displayW_, displayH_);
        activeMode_ = mode_ == UpscaleMode::Auto ? autoMode(displayW_, displayH_) : mode_;
        const float s = upscale_ ? fsrScale(activeMode_) : 1.0f;
        motionW_ = std::max(1u, static_cast<uint32_t>(std::lround(displayW_ / s)));
        motionH_ = std::max(1u, static_cast<uint32_t>(std::lround(displayH_ / s)));
        presentValid_ = false;
        showingMotion_ = true;
        CRTX_LOG("Vulkan : rendu %ux%u en mouvement%s, natif %ux%u à l'arrêt", motionW_, motionH_,
                 motionW_ != displayW_ ? " (AMD FSR 1 EASU + RCAS)" : "", displayW_, displayH_);
    }

    void waitIdle() override
    {
        vk_->waitTimeline(frameCount_);
        vkDeviceWaitIdle(vk_->device());
    }

    bool frame(const FrameContext& ctx) override;
    std::string stats(float fps) const override;
    bool saveScreenshot(const std::string& path) override;

private:
    GLFWwindow* window_ = nullptr;
    BackendOptions opt_;
    std::unique_ptr<VulkanContext> vk_;
    std::unique_ptr<calc::ICalcEngine> calc_;
    std::unique_ptr<VkRenderer> renderer_;
    VkCommandBuffer cmd_ = VK_NULL_HANDLE;
    bool fp64_ = true;

    uint32_t displayW_ = 0, displayH_ = 0, motionW_ = 0, motionH_ = 0;
    bool upscale_ = true;
    UpscaleMode mode_ = UpscaleMode::Auto;
    UpscaleMode activeMode_ = UpscaleMode::Quality;
    uint64_t frameCount_ = 0;
    int hovered_ = -1;
    bool idle_ = false;
    bool presentValid_ = false;
    bool showingMotion_ = true;
    unsigned staticSpp_ = 1;
    const char* lastJob_ = "";
};

bool VulkanBackend::frame(const FrameContext& ctx)
{
    vk_->waitTimeline(frameCount_);  // l'image précédente est terminée
    const uint64_t signal = frameCount_ + 1;
    renderer_->beginFrame();
    hovered_ = ctx.hoverEnabled ? renderer_->pickResult() : -1;

    if (ctx.changed) renderer_->resetAccumulation();
    const bool motionFrame = ctx.changed;
    const unsigned acc = renderer_->accumulated();
    const bool converged = !motionFrame && acc > 0 && renderer_->renderWidth() == displayW_ &&
                           acc * staticSpp_ >= ctx.settings.convergedSamples;
    idle_ = converged;

    VkFrameDesc f;
    f.camera = ctx.camera;
    f.prevCamera = ctx.prevCamera;
    f.frameIndex = static_cast<uint32_t>(ctx.frameIndex);
    f.maxBounces = ctx.settings.maxBounces;
    f.fireflyClamp = ctx.settings.fireflyClamp;
    f.denoise = ctx.settings.denoise;
    f.exposure = ctx.post.exposure;
    f.bloomStrength = ctx.post.bloomStrength;
    f.bloomThreshold = ctx.post.bloomThreshold;
    f.vignette = ctx.post.vignette;
    f.bgra = window_ ? vk_->swapIsBgra() : true;
    if (motionFrame) {
        f.renderW = motionW_;
        f.renderH = motionH_;
        f.spp = ctx.settings.spp;
        f.resetAccumulation = true;
        f.temporal = true;
    } else {
        f.renderW = displayW_;
        f.renderH = displayH_;
        f.spp = staticSpp_;
        f.resetAccumulation = renderer_->renderWidth() != displayW_;
    }
    f.pickX = ctx.cursorU * static_cast<float>(f.renderW);
    f.pickY = ctx.cursorV * static_cast<float>(f.renderH);
    // L'image native ne remplace l'image FSR qu'après quelques images accumulées
    const bool upscaledMotion = motionW_ != displayW_;
    f.runPost = motionFrame || !upscaledMotion || !showingMotion_ || acc + 1 >= kShowStatic;

    VK_CHECK(vkResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
    renderer_->syncScene(*ctx.scene, cmd_);
    bool newImage = false;
    if (converged) {
        f.pickX = ctx.cursorU * static_cast<float>(renderer_->renderWidth());
        f.pickY = ctx.cursorV * static_cast<float>(renderer_->renderHeight());
        renderer_->recordPick(cmd_, f);
        lastJob_ = "convergé";
    } else {
        renderer_->recordFrame(cmd_, f);
        newImage = f.runPost;
        if (newImage) showingMotion_ = motionFrame && upscaledMotion;
        lastJob_ = motionFrame ? (upscaledMotion ? "FSR 1" : "natif") : "accumulation";
    }

    bool ok = true;
    if (window_ && newImage) {
        uint32_t index = 0;
        const VkResult ar = vk_->acquireImage(index);
        if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
            VK_CHECK(vkEndCommandBuffer(cmd_));
            VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
            ts.signalSemaphoreValueCount = 1;
            ts.pSignalSemaphoreValues = &signal;
            const VkSemaphore tl = vk_->timeline();
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.pNext = &ts;
            si.commandBufferCount = 1;
            si.pCommandBuffers = &cmd_;
            si.signalSemaphoreCount = 1;
            si.pSignalSemaphores = &tl;
            VK_CHECK(vkQueueSubmit(vk_->queue(), 1, &si, VK_NULL_HANDLE));
            ok = false;
        } else {
            vk_->recordCopyToSwapchain(cmd_, renderer_->presentBuffer().buffer, index);
            VK_CHECK(vkEndCommandBuffer(cmd_));
            ok = vk_->submitAndPresent(cmd_, index, 0, signal) && ar != VK_SUBOPTIMAL_KHR;
        }
    } else {
        VK_CHECK(vkEndCommandBuffer(cmd_));
        VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ts.signalSemaphoreValueCount = 1;
        ts.pSignalSemaphoreValues = &signal;
        const VkSemaphore tl = vk_->timeline();
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &ts;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd_;
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &tl;
        VK_CHECK(vkQueueSubmit(vk_->queue(), 1, &si, VK_NULL_HANDLE));
    }
    if (newImage) presentValid_ = true;
    // échantillons par image immobile : vise ~12 ms de path tracing
    const float ms = renderer_->lastTraceMs();
    if (!motionFrame && !converged && ms > 0.0f) {
        const float perSample = ms / static_cast<float>(std::max(1u, staticSpp_));
        staticSpp_ = std::clamp(static_cast<unsigned>(12.0f / std::max(perSample, 0.1f)), 1u, 8u);
    }
    ++frameCount_;
    return ok;
}

std::string VulkanBackend::stats(float fps) const
{
    char buf[512];
    std::snprintf(buf, sizeof(buf),
                  "CalculatoRTX | %s | Vulkan RT%s %ux%u -> %ux%u | %.0f ips | %s | PT %.2f ms | post %.2f ms | %u éch. | calcul %s",
                  vk_->deviceName().c_str(), motionW_ != displayW_ ? " + FSR 1" : "", renderer_->renderWidth(),
                  renderer_->renderHeight(), displayW_, displayH_, fps, lastJob_, renderer_->lastTraceMs(),
                  renderer_->lastPostMs(), renderer_->accumulated() * staticSpp_, calc_->apiName());
    return buf;
}

bool VulkanBackend::saveScreenshot(const std::string& path)
{
    waitIdle();
    if (!presentValid_) return false;
    const Buffer& src = renderer_->presentBuffer();
    const VkDeviceSize bytes = static_cast<VkDeviceSize>(displayW_) * displayH_ * 4;
    Buffer staging = vk_->createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    vk_->immediateSubmit([&](VkCommandBuffer cmd) {
        const VkBufferCopy region{0, 0, bytes};
        vkCmdCopyBuffer(cmd, src.buffer, staging.buffer, 1, &region);
        VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                     VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    });
    std::vector<uint32_t> px(static_cast<size_t>(displayW_) * displayH_);
    std::memcpy(px.data(), staging.mapped, static_cast<size_t>(bytes));
    vk_->destroyBuffer(staging);
    return writeBmp(path, px, displayW_, displayH_, window_ ? vk_->swapIsBgra() : true);
}

}  // namespace

std::unique_ptr<Backend> createVulkanBackend(GLFWwindow* window, const BackendOptions& opt)
{
    return std::make_unique<VulkanBackend>(window, opt);
}

const char* upscaleModeName(UpscaleMode m)
{
    switch (m) {
        case UpscaleMode::Auto: return "auto";
        case UpscaleMode::Native: return "natif";
        case UpscaleMode::Quality: return "Qualité";
        case UpscaleMode::Balanced: return "Équilibré";
        case UpscaleMode::Performance: return "Performance";
        case UpscaleMode::UltraPerformance: return "Ultra Performance";
    }
    return "?";
}

}  // namespace crtx
