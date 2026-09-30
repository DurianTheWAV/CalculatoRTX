// CalculatoRTX - backend NVIDIA : CUDA + OptiX + DLSS + débruiteur IA (voir NvidiaBackend.h).
#include "NvidiaBackend.h"

#include "../app/Screenshot.h"
#include "../calc/CalcEngine.h"
#include "../dlss/DlssUpscaler.h"
#include "../gpu/VulkanContext.h"
#include "../render/EmbeddedPtx.h"
#include "../render/Kernels.h"
#include "../render/OptixRenderer.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>

namespace crtx {

namespace fs = std::filesystem;

namespace {

float halton(unsigned index, unsigned base)
{
    float f = 1.0f, r = 0.0f;
    while (index > 0) {
        f /= static_cast<float>(base);
        r += f * static_cast<float>(index % base);
        index /= base;
    }
    return r;
}

fs::path executableDir()
{
#if defined(_WIN32)
    wchar_t buf[MAX_PATH] = {};
    GetModuleFileNameW(nullptr, buf, MAX_PATH);
    return fs::path(buf).parent_path();
#else
    std::error_code ec;
    const fs::path p = fs::read_symlink("/proc/self/exe", ec);
    return ec ? fs::current_path() : p.parent_path();
#endif
}

DlssMode toDlss(UpscaleMode m)
{
    switch (m) {
        case UpscaleMode::Native: return DlssMode::Dlaa;
        case UpscaleMode::Balanced: return DlssMode::Balanced;
        case UpscaleMode::Performance: return DlssMode::Performance;
        case UpscaleMode::UltraPerformance: return DlssMode::UltraPerformance;
        default: return DlssMode::Quality;
    }
}

// Mode DLSS automatique : fonction de la définition d'affichage et de la taille du GPU
// (RTX 4060 Ti : 34 SM -> Quality jusqu'au 1080p, Balanced en 1440p, Performance en 4K).
UpscaleMode autoMode(uint32_t w, uint32_t h, int smCount)
{
    const double mp = static_cast<double>(w) * h / 1.0e6;
    int level = mp <= 2.2 ? 0 : (mp <= 3.8 ? 1 : 2);  // 0 Quality, 1 Balanced, 2 Performance
    if (smCount >= 56 && level > 0) --level;           // RTX 4070 Ti et plus
    if (smCount > 0 && smCount < 30 && level < 2) ++level;  // RTX 4060 et moins
    return level == 0 ? UpscaleMode::Quality : level == 1 ? UpscaleMode::Balanced : UpscaleMode::Performance;
}

constexpr unsigned kShowStatic = 4;  // images natives accumulées avant de remplacer l'image DLSS

class NvidiaBackend final : public Backend {
public:
    NvidiaBackend(GLFWwindow* window, const BackendOptions& opt) : window_(window), opt_(opt)
    {
        if (const char* js = std::getenv("CRTX_DLSS_JITTER_SIGN")) jitterSign_ = std::atof(js) < 0.0 ? -1.0f : 1.0f;
        mode_ = opt.mode;
        upscale_ = opt.upscale;
        initCuda();
        calc_ = std::make_unique<calc::CalcEngine>();
        initVulkanAndDlss();
        renderer_ = std::make_unique<OptixRenderer>();
        std::vector<unsigned char> ptx(g_optixProgramsPtx, g_optixProgramsPtx + g_optixProgramsPtxSize);
        renderer_->init(ptx);
        post_ = std::make_unique<PostProcessor>();
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = vk_->commandPool();
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(vk_->device(), &ai, &cmdDlss_));
        VK_CHECK(vkAllocateCommandBuffers(vk_->device(), &ai, &cmdPresent_));
    }

    ~NvidiaBackend() override
    {
        if (vk_ && vk_->device()) vkDeviceWaitIdle(vk_->device());
        if (stream_) cudaStreamSynchronize(stream_);
        releaseFrameResources();
        if (dlss_) dlss_->shutdown();
        post_.reset();
        renderer_.reset();
        calc_.reset();
        dlss_.reset();
        if (vk_) {
            if (cmdDlss_) vkFreeCommandBuffers(vk_->device(), vk_->commandPool(), 1, &cmdDlss_);
            if (cmdPresent_) vkFreeCommandBuffers(vk_->device(), vk_->commandPool(), 1, &cmdPresent_);
        }
        vk_.reset();
        if (stream_) cudaStreamDestroy(stream_);
    }

    std::string name() const override { return "NVIDIA CUDA/OptiX"; }
    std::string deviceName() const override { return gpuName_; }
    calc::ICalcEngine& calc() override { return *calc_; }
    int hoveredKey() const override { return hovered_; }
    bool idle() const override { return idle_; }
    float aspect() const override { return static_cast<float>(displayW_) / static_cast<float>(std::max(displayH_, 1u)); }
    std::string upscalerName() const override { return "DLSS"; }
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
        vk_->waitTimeline(frameCount_ * 4);
        vkDeviceWaitIdle(vk_->device());
        CUDA_CHECK(cudaStreamSynchronize(stream_));
        vk_->createSwapchain(fbw, fbh, opt_.vsync);
        configure();
    }

    void waitIdle() override
    {
        vk_->waitTimeline(frameCount_ * 4);
        vkDeviceWaitIdle(vk_->device());
        cudaStreamSynchronize(stream_);
    }

    bool frame(const FrameContext& ctx) override;
    std::string stats(float fps) const override;
    bool saveScreenshot(const std::string& path) override;

private:
    void initCuda();
    void initVulkanAndDlss();
    void configure();
    void releaseFrameResources();
    void cudaSignal(uint64_t value);
    void cudaWait(uint64_t value);
    void evaluateDlss(uint64_t base, const float2& jitter, bool reset);
    bool present(uint64_t base, bool newImage);

    GLFWwindow* window_ = nullptr;
    BackendOptions opt_;
    cudaStream_t stream_ = nullptr;
    unsigned char cudaUuid_[16] = {};
    std::string gpuName_;
    int smCount_ = 0;

    std::unique_ptr<VulkanContext> vk_;
    std::unique_ptr<DlssUpscaler> dlss_;
    std::unique_ptr<calc::CalcEngine> calc_;
    std::unique_ptr<OptixRenderer> renderer_;
    std::unique_ptr<PostProcessor> post_;

    uint32_t displayW_ = 0, displayH_ = 0, renderW_ = 0, renderH_ = 0;
    bool upscale_ = true;
    UpscaleMode mode_ = UpscaleMode::Auto;
    UpscaleMode activeMode_ = UpscaleMode::Quality;
    bool dlssActive_ = false;
    SharedBuffer shColor_, shDepth_, shMotion_, shDlssOut_, shPresent_;
    GpuImage imgColor_, imgDepth_, imgMotion_, imgOut_;
    VkCommandBuffer cmdDlss_ = VK_NULL_HANDLE, cmdPresent_ = VK_NULL_HANDLE;
    unsigned jitterPhases_ = 8;
    unsigned jitterIndex_ = 0;
    float jitterSign_ = 1.0f;

    uint64_t frameCount_ = 0;
    int hovered_ = -1;
    bool idle_ = false;
    bool presentValid_ = false;      // shPresent_ contient une image terminée
    bool showingMotion_ = true;      // l'image affichée provient du rendu en mouvement (DLSS)
    bool resetDlss_ = true;
    CameraData prevDlssCamera_{};
    unsigned staticSpp_ = 2;
    float lastFrameMs_ = 16.0f;
    const char* lastJob_ = "";
    bool ser_ = true;
};

void NvidiaBackend::initCuda()
{
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) {
        cudaGetLastError();
        throwError("Aucun GPU CUDA utilisable (pilote NVIDIA absent ou trop ancien)", __FILE__, __LINE__);
    }
    const int dev = std::min(std::max(opt_.device, 0), count - 1);
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::memcpy(cudaUuid_, prop.uuid.bytes, 16);
    gpuName_ = prop.name;
    smCount_ = prop.multiProcessorCount;
    int clockKHz = 0;
    cudaDeviceGetAttribute(&clockKHz, cudaDevAttrClockRate, dev);
    const bool ada = prop.major == 8 && prop.minor == 9;
    CRTX_LOG("GPU CUDA : %s | sm_%d%d%s | %d SM | %.1f Go | L2 %.0f Mo | %.2f GHz", prop.name, prop.major, prop.minor,
             ada ? " (Ada Lovelace)" : "", prop.multiProcessorCount, prop.totalGlobalMem / 1073741824.0,
             prop.l2CacheSize / 1048576.0, clockKHz / 1.0e6);
    if (prop.major < 7 || (prop.major == 7 && prop.minor < 5))
        CRTX_LOG("Attention : GPU sans RT cores ni Tensor cores, performances très réduites");
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
}

void NvidiaBackend::initVulkanAndDlss()
{
    vk_ = std::make_unique<VulkanContext>();
    dlss_ = std::make_unique<DlssUpscaler>();
    const bool wantDlss = DlssUpscaler::compiledIn();
    vk_->createInstance(wantDlss ? DlssUpscaler::requiredInstanceExtensions() : std::vector<std::string>{},
                        opt_.validation);
    vk_->pickPhysicalDeviceByUuid(cudaUuid_);
    DeviceRequest req;
    req.cudaInterop = true;
    if (wantDlss) req.extraExtensions = DlssUpscaler::requiredDeviceExtensions(vk_->instance(), vk_->physicalDevice());
    vk_->createDevice(window_, req);

    if (wantDlss) {
        const fs::path exeDir = executableDir();
        std::error_code ec;
        const fs::path dataDir = fs::temp_directory_path(ec) / "CalculatoRTX";
        fs::create_directories(dataDir, ec);
        dlss_->init(vk_->instance(), vk_->physicalDevice(), vk_->device(), dataDir.wstring(), exeDir.wstring());
    } else {
        dlss_->init(VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE, L"", L"");
    }
    CRTX_LOG("%s", dlss_->status().c_str());
}

void NvidiaBackend::releaseFrameResources()
{
    if (!vk_) return;
    if (dlss_) dlss_->releaseFeature();
    for (SharedBuffer* b : {&shColor_, &shDepth_, &shMotion_, &shDlssOut_, &shPresent_})
        if (b->buffer) vk_->destroySharedBuffer(*b);
    for (GpuImage* i : {&imgColor_, &imgDepth_, &imgMotion_, &imgOut_})
        if (i->image) vk_->destroyImage(*i);
}

void NvidiaBackend::configure()
{
    releaseFrameResources();
    displayW_ = vk_->swapExtent().width;
    displayH_ = vk_->swapExtent().height;
    activeMode_ = mode_ == UpscaleMode::Auto ? autoMode(displayW_, displayH_, smCount_) : mode_;
    const DlssMode dm = toDlss(activeMode_);
    dlssActive_ = upscale_ && dlss_->available();
    if (dlssActive_ && !dlss_->optimalRenderSize(dm, displayW_, displayH_, renderW_, renderH_)) {
        CRTX_LOG("DLSS : mode %s indisponible à cette résolution", dlssModeName(dm));
        dlssActive_ = false;
    }
    shPresent_ = vk_->createSharedBuffer(static_cast<VkDeviceSize>(displayW_) * displayH_ * 4);

    if (dlssActive_) {
        const VkDeviceSize rn = static_cast<VkDeviceSize>(renderW_) * renderH_;
        shColor_ = vk_->createSharedBuffer(rn * 8);
        shDepth_ = vk_->createSharedBuffer(rn * 4);
        shMotion_ = vk_->createSharedBuffer(rn * 8);
        shDlssOut_ = vk_->createSharedBuffer(static_cast<VkDeviceSize>(displayW_) * displayH_ * 8);
        const VkImageUsageFlags in = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
        imgColor_ = vk_->createImage(renderW_, renderH_, VK_FORMAT_R16G16B16A16_SFLOAT, in);
        imgDepth_ = vk_->createImage(renderW_, renderH_, VK_FORMAT_R32_SFLOAT, in);
        imgMotion_ = vk_->createImage(renderW_, renderH_, VK_FORMAT_R32G32_SFLOAT, in);
        imgOut_ = vk_->createImage(displayW_, displayH_, VK_FORMAT_R16G16B16A16_SFLOAT,
                                   VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
        bool ok = false;
        vk_->immediateSubmit([&](VkCommandBuffer cmd) {
            ok = dlss_->createFeature(cmd, dm, renderW_, renderH_, displayW_, displayH_);
        });
        if (!ok) {
            CRTX_LOG("%s -> rendu natif", dlss_->status().c_str());
            for (SharedBuffer* b : {&shColor_, &shDepth_, &shMotion_, &shDlssOut_}) vk_->destroySharedBuffer(*b);
            for (GpuImage* i : {&imgColor_, &imgDepth_, &imgMotion_, &imgOut_}) vk_->destroyImage(*i);
            dlssActive_ = false;
        }
    }
    renderer_->resize(kTargetMotion, dlssActive_ ? renderW_ : 0, dlssActive_ ? renderH_ : 0);
    renderer_->resize(kTargetStatic, displayW_, displayH_);
    renderer_->resetAccumulation(kTargetStatic);
    if (!dlssActive_) {
        renderW_ = displayW_;
        renderH_ = displayH_;
    }
    const float ratio = static_cast<float>(displayW_) / static_cast<float>(renderW_);
    jitterPhases_ = std::max(8u, static_cast<unsigned>(std::lround(8.0f * ratio * ratio)));
    resetDlss_ = true;
    presentValid_ = false;
    showingMotion_ = true;
    if (dlssActive_)
        CRTX_LOG("DLSS %s : rendu %ux%u -> affichage %ux%u en mouvement, natif %ux%u à l'arrêt (%u phases de jitter)",
                 dlssModeName(dm), renderW_, renderH_, displayW_, displayH_, displayW_, displayH_, jitterPhases_);
    else
        CRTX_LOG("Rendu natif %ux%u (accumulation progressive + débruiteur IA OptiX)", displayW_, displayH_);
}

void NvidiaBackend::cudaSignal(uint64_t value)
{
    cudaExternalSemaphore_t sem = vk_->cudaTimeline();
    cudaExternalSemaphoreSignalParams p{};
    p.params.fence.value = value;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&sem, &p, 1, stream_));
}

void NvidiaBackend::cudaWait(uint64_t value)
{
    cudaExternalSemaphore_t sem = vk_->cudaTimeline();
    cudaExternalSemaphoreWaitParams p{};
    p.params.fence.value = value;
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&sem, &p, 1, stream_));
}

// Vulkan : copies des entrées -> DLSS (Tensor cores) -> copie de la sortie ; timeline base+1 -> base+2
void NvidiaBackend::evaluateDlss(uint64_t base, const float2& jitter, bool reset)
{
    VK_CHECK(vkResetCommandBuffer(cmdDlss_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmdDlss_, &bi));
    const std::pair<SharedBuffer*, GpuImage*> inputs[] = {{&shColor_, &imgColor_}, {&shDepth_, &imgDepth_}, {&shMotion_, &imgMotion_}};
    for (const auto& io : inputs) {
        VulkanContext::imageBarrier(cmdDlss_, io.second->image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {renderW_, renderH_, 1};
        vkCmdCopyBufferToImage(cmdDlss_, io.first->buffer, io.second->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VulkanContext::imageBarrier(cmdDlss_, io.second->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_SHADER_READ_BIT);
    }
    VulkanContext::imageBarrier(cmdDlss_, imgOut_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    const DlssImages images{imgColor_.image, imgColor_.view, imgColor_.format, imgDepth_.image, imgDepth_.view,
                            imgDepth_.format, imgMotion_.image, imgMotion_.view, imgMotion_.format, imgOut_.image,
                            imgOut_.view, imgOut_.format};
    if (!dlss_->evaluate(cmdDlss_, images, renderW_, renderH_, displayW_, displayH_, jitter.x, jitter.y, reset,
                         lastFrameMs_))
        CRTX_LOG("DLSS : échec de l'évaluation");
    VulkanContext::imageBarrier(cmdDlss_, imgOut_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    VkBufferImageCopy outRegion{};
    outRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    outRegion.imageExtent = {displayW_, displayH_, 1};
    vkCmdCopyImageToBuffer(cmdDlss_, imgOut_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shDlssOut_.buffer, 1, &outRegion);
    VK_CHECK(vkEndCommandBuffer(cmdDlss_));

    const VkSemaphore tl = vk_->timeline();
    const uint64_t waitV = base + 1, sigV = base + 2;
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.waitSemaphoreValueCount = 1;
    ts.pWaitSemaphoreValues = &waitV;
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues = &sigV;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ts;
    si.waitSemaphoreCount = 1;
    si.pWaitSemaphores = &tl;
    si.pWaitDstStageMask = &stage;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmdDlss_;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &tl;
    VK_CHECK(vkQueueSubmit(vk_->queue(), 1, &si, VK_NULL_HANDLE));
}

// Vulkan : attend base+3 (CUDA terminé), copie le tampon vers la swapchain, présente, signale base+4.
bool NvidiaBackend::present(uint64_t base, bool newImage)
{
    if (!newImage && !presentValid_) {
        vk_->signalTimeline(base + 3, base + 4);
        return true;
    }
    uint32_t index = 0;
    const VkResult ar = vk_->acquireImage(index);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
        vk_->signalTimeline(base + 3, base + 4);  // la timeline reste monotone
        return false;
    }
    VK_CHECK(vkResetCommandBuffer(cmdPresent_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmdPresent_, &bi));
    vk_->recordCopyToSwapchain(cmdPresent_, shPresent_.buffer, index);
    VK_CHECK(vkEndCommandBuffer(cmdPresent_));
    const bool ok = vk_->submitAndPresent(cmdPresent_, index, base + 3, base + 4);
    return ok && ar != VK_SUBOPTIMAL_KHR;
}

bool NvidiaBackend::frame(const FrameContext& ctx)
{
    const uint64_t base = frameCount_ * 4;
    vk_->waitTimeline(base);  // l'image précédente est entièrement terminée (CPU <-> GPU)
    lastFrameMs_ = ctx.dt * 1000.0f;
    ser_ = ctx.settings.ser;
    hovered_ =ctx.hoverEnabled ? renderer_->pickResult() : -1;

    renderer_->syncScene(*ctx.scene, stream_);
    if (ctx.changed) renderer_->resetAccumulation(kTargetStatic);
    const unsigned staticAccum = renderer_->accumulatedFrames(kTargetStatic);
    const bool converged = !ctx.changed && staticAccum > 0 && staticAccum * staticSpp_ >= ctx.settings.convergedSamples;

    // ---- choix de l'image à calculer
    enum class Job { None, Dlss, Native };
    Job job = Job::None;
    if (ctx.changed) job = dlssActive_ ? Job::Dlss : Job::Native;
    else if (!converged) job = Job::Native;
    idle_ = converged;

    const int target = job == Job::Dlss ? kTargetMotion : kTargetStatic;
    const float tw = static_cast<float>(renderer_->width(target)), th = static_cast<float>(renderer_->height(target));
    const bool bgra = vk_->swapIsBgra();
    PostProcessor::Params pp;
    pp.exposure = ctx.post.exposure;
    pp.bloomStrength = ctx.post.bloomStrength;
    pp.bloomThreshold = ctx.post.bloomThreshold;
    pp.vignette = ctx.post.vignette;
    pp.frame = static_cast<unsigned>(ctx.frameIndex);
    bool newImage = false;

    if (job == Job::Dlss) {
        FrameInput in;
        in.camera = ctx.camera;
        in.prevCamera = resetDlss_ ? ctx.camera : prevDlssCamera_;
        in.dlss = true;
        in.frameIndex = static_cast<unsigned>(ctx.frameIndex);
        in.time = ctx.time;
        const unsigned idx = (jitterIndex_++ % jitterPhases_) + 1;
        in.jitter = make_float2(halton(idx, 2) - 0.5f, halton(idx, 3) - 0.5f);
        RenderSettings rs = ctx.settings;
        const float4* hdr = renderer_->render(kTargetMotion, in, rs, stream_);
        renderer_->pick(kTargetMotion, ctx.camera, make_float2(ctx.cursorU * tw, ctx.cursorV * th), stream_);
        const size_t rn = static_cast<size_t>(renderW_) * renderH_;
        kernels::packHalf4(hdr, static_cast<__half*>(shColor_.cudaPtr), static_cast<int>(rn), stream_);
        CUDA_CHECK(cudaMemcpyAsync(shDepth_.cudaPtr, renderer_->depthBuffer(kTargetMotion), rn * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(shMotion_.cudaPtr, renderer_->motionBuffer(kTargetMotion), rn * sizeof(float2),
                                   cudaMemcpyDeviceToDevice, stream_));
        cudaSignal(base + 1);
        // Jitter transmis au DLSS (CRTX_DLSS_JITTER_SIGN=-1 inverse la convention, pour diagnostic)
        evaluateDlss(base, make_float2(in.jitter.x * jitterSign_, in.jitter.y * jitterSign_), resetDlss_);
        cudaWait(base + 2);
        post_->run(shDlssOut_.cudaPtr, true, static_cast<unsigned char*>(shPresent_.cudaPtr), bgra,
                   static_cast<int>(displayW_), static_cast<int>(displayH_), pp, stream_);
        prevDlssCamera_ = ctx.camera;
        resetDlss_ = false;
        showingMotion_ = true;
        newImage = true;
        lastJob_ = "DLSS";
    } else if (job == Job::Native) {
        FrameInput in;
        in.camera = ctx.camera;
        in.prevCamera = ctx.prevCamera;
        in.dlss = false;
        in.resetAccumulation = ctx.changed;
        in.frameIndex = static_cast<unsigned>(ctx.frameIndex);
        in.time = ctx.time;
        RenderSettings rs = ctx.settings;
        // en mouvement sans DLSS : 1 image = settings.spp ; immobile : échantillons adaptés au GPU
        if (!ctx.changed) rs.spp = staticSpp_;
        const float4* hdr = renderer_->render(kTargetStatic, in, rs, stream_);
        renderer_->pick(kTargetStatic, ctx.camera, make_float2(ctx.cursorU * tw, ctx.cursorV * th), stream_);
        // Avec DLSS, l'image native ne remplace l'image DLSS qu'une fois quelques images accumulées
        const bool show = !dlssActive_ || ctx.changed || renderer_->accumulatedFrames(kTargetStatic) >= kShowStatic ||
                          !showingMotion_;
        if (show) {
            post_->run(hdr, false, static_cast<unsigned char*>(shPresent_.cudaPtr), bgra, static_cast<int>(displayW_),
                       static_cast<int>(displayH_), pp, stream_);
            showingMotion_ = false;
            newImage = true;
        }
        lastJob_ = ctx.changed ? "natif" : "accumulation";
        // échantillons par image immobile : vise ~12 ms de path tracing (RTX 4060 Ti : 2 à 4)
        const float ms = renderer_->lastTraceMs();
        if (!ctx.changed && ms > 0.0f) {
            const float perSample = ms / static_cast<float>(std::max(1u, staticSpp_));
            staticSpp_ = std::clamp(static_cast<unsigned>(12.0f / std::max(perSample, 0.1f)), 1u, 8u);
        }
    } else {
        // image convergée : seul le rayon de sélection est lancé (survol des touches)
        renderer_->pick(kTargetStatic, ctx.camera, make_float2(ctx.cursorU * tw, ctx.cursorV * th), stream_);
        lastJob_ = "convergé";
    }
    cudaSignal(base + 3);
    const bool ok = present(base, newImage);
    if (newImage) presentValid_ = true;
    ++frameCount_;
    return ok;
}

std::string NvidiaBackend::stats(float fps) const
{
    char buf[512];
    const unsigned acc = renderer_->accumulatedFrames(kTargetStatic) * staticSpp_;
    if (dlssActive_)
        std::snprintf(buf, sizeof(buf),
                      "CalculatoRTX | %s | OptiX + DLSS %s %ux%u -> %ux%u | %.0f ips | %s | PT %.2f ms | IA %.2f ms | %u éch.%s",
                      gpuName_.c_str(), upscaleModeName(activeMode_), renderW_, renderH_, displayW_, displayH_, fps,
                      lastJob_, renderer_->lastTraceMs(), renderer_->lastDenoiseMs(), acc, ser_ ? " | SER" : "");
    else
        std::snprintf(buf, sizeof(buf), "CalculatoRTX | %s | OptiX natif %ux%u | %.0f ips | %s | PT %.2f ms | IA %.2f ms | %u éch.%s",
                      gpuName_.c_str(), displayW_, displayH_, fps, lastJob_, renderer_->lastTraceMs(),
                      renderer_->lastDenoiseMs(), acc, ser_ ? " | SER" : "");
    return buf;
}

bool NvidiaBackend::saveScreenshot(const std::string& path)
{
    waitIdle();
    if (!presentValid_) return false;
    std::vector<uint32_t> px(static_cast<size_t>(displayW_) * displayH_);
    CUDA_CHECK(cudaMemcpy(px.data(), shPresent_.cudaPtr, px.size() * 4, cudaMemcpyDeviceToHost));
    return writeBmp(path, px, displayW_, displayH_, vk_->swapIsBgra());
}

}  // namespace

std::unique_ptr<Backend> createNvidiaBackend(GLFWwindow* window, const BackendOptions& opt)
{
    return std::make_unique<NvidiaBackend>(window, opt);
}

}  // namespace crtx
