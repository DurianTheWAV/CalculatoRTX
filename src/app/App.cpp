// CalculatoRTX - boucle principale.
//
// Chronologie d'une image (valeurs du sémaphore timeline partagé CUDA/Vulkan, base = 4*n) :
//   CUDA   : path tracing OptiX -> débruiteur IA -> copie des entrées DLSS -> signal(base+1)
//   Vulkan : attend(base+1) -> copies tampon->image -> DLSS -> copie image->tampon -> signal(base+2)
//   CUDA   : attend(base+2) -> post-traitement (CUDA Graph) -> signal(base+3)
//   Vulkan : attend(base+3) -> copie vers la swapchain -> présentation -> signal(base+4)
// Sans DLSS, les étapes base+1/base+2 sont sautées (valeurs monotones, donc valide).
#include "App.h"

#include "../render/EmbeddedPtx.h"

#include <algorithm>
#include <chrono>
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

App* appFrom(GLFWwindow* w) { return static_cast<App*>(glfwGetWindowUserPointer(w)); }

}  // namespace

App::App(const AppOptions& opt) : opt_(opt)
{
    if (const char* js = std::getenv("CRTX_DLSS_JITTER_SIGN")) jitterSign_ = std::atof(js) < 0.0 ? -1.0f : 1.0f;
    initCuda();
    engine_ = std::make_unique<calc::CalcEngine>();
    if (calc::runSelfTest(*engine_)) CRTX_LOG("Moteur de calcul GPU : auto-test OK (double-double / FP64 / FP32 / intervalles)");
    controller_ = std::make_unique<CalculatorController>(*engine_);

    initWindow();
    initVulkanAndDlss();

    scene_ = std::make_unique<CalculatorScene>(font_);
    scene_->build();
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

    int fbw = 0, fbh = 0;
    glfwGetFramebufferSize(window_, &fbw, &fbh);
    vk_->createSwapchain(static_cast<uint32_t>(fbw), static_cast<uint32_t>(fbh), opt_.vsync);
    configureResolution();
    orbitTarget_ = orbit_;
    printHelp();
}

App::~App()
{
    if (vk_ && vk_->device()) vkDeviceWaitIdle(vk_->device());
    if (stream_) cudaStreamSynchronize(stream_);
    releaseFrameResources();
    if (dlss_) dlss_->shutdown();
    post_.reset();
    renderer_.reset();
    scene_.reset();
    controller_.reset();
    engine_.reset();
    dlss_.reset();
    if (vk_) {
        if (cmdDlss_) vkFreeCommandBuffers(vk_->device(), vk_->commandPool(), 1, &cmdDlss_);
        if (cmdPresent_) vkFreeCommandBuffers(vk_->device(), vk_->commandPool(), 1, &cmdPresent_);
    }
    vk_.reset();
    if (stream_) cudaStreamDestroy(stream_);
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

void App::initCuda()
{
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count == 0) throwError("Aucun GPU CUDA détecté", __FILE__, __LINE__);
    const int dev = std::min(std::max(opt_.cudaDevice, 0), count - 1);
    CUDA_CHECK(cudaSetDevice(dev));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::memcpy(cudaUuid_, prop.uuid.bytes, 16);
    gpuName_ = prop.name;
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

void App::initWindow()
{
    if (!glfwInit()) throwError("Échec d'initialisation de GLFW", __FILE__, __LINE__);
    if (!glfwVulkanSupported()) throwError("Vulkan indisponible (pilote NVIDIA / chargeur Vulkan ?)", __FILE__, __LINE__);
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);
    window_ = glfwCreateWindow(opt_.width, opt_.height, "CalculatoRTX", nullptr, nullptr);
    if (!window_) throwError("Impossible de créer la fenêtre", __FILE__, __LINE__);
    glfwSetWindowUserPointer(window_, this);
    glfwSetCharCallback(window_, [](GLFWwindow* w, unsigned int c) { appFrom(w)->onKeyChar(c); });
    glfwSetKeyCallback(window_, [](GLFWwindow* w, int key, int, int action, int mods) { appFrom(w)->onKey(key, action, mods); });
    glfwSetMouseButtonCallback(window_, [](GLFWwindow* w, int b, int action, int) { appFrom(w)->onMouseButton(b, action); });
    glfwSetCursorPosCallback(window_, [](GLFWwindow* w, double x, double y) { appFrom(w)->onCursor(x, y); });
    glfwSetScrollCallback(window_, [](GLFWwindow* w, double, double dy) { appFrom(w)->onScroll(dy); });
    glfwSetFramebufferSizeCallback(window_, [](GLFWwindow* w, int, int) { appFrom(w)->swapchainDirty_ = true; });
}

void App::initVulkanAndDlss()
{
    vk_ = std::make_unique<VulkanContext>();
    dlss_ = std::make_unique<DlssUpscaler>();
    const bool wantDlss = DlssUpscaler::compiledIn();
    vk_->createInstance(wantDlss ? DlssUpscaler::requiredInstanceExtensions() : std::vector<std::string>{},
                        opt_.validation);
    vk_->pickPhysicalDevice(cudaUuid_);
    const std::vector<std::string> devExt =
        wantDlss ? DlssUpscaler::requiredDeviceExtensions(vk_->instance(), vk_->physicalDevice()) : std::vector<std::string>{};
    vk_->createDevice(window_, devExt);

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

void App::releaseFrameResources()
{
    if (!vk_) return;
    if (dlss_) dlss_->releaseFeature();
    for (SharedBuffer* b : {&shColor_, &shDepth_, &shMotion_, &shDlssOut_, &shPresent_})
        if (b->buffer) vk_->destroySharedBuffer(*b);
    for (GpuImage* i : {&imgColor_, &imgDepth_, &imgMotion_, &imgOut_})
        if (i->image) vk_->destroyImage(*i);
}

void App::configureResolution()
{
    vkDeviceWaitIdle(vk_->device());
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    releaseFrameResources();

    displayW_ = vk_->swapExtent().width;
    displayH_ = vk_->swapExtent().height;
    dlssActive_ = opt_.dlss && dlss_->available();
    if (dlssActive_ && !dlss_->optimalRenderSize(opt_.dlssMode, displayW_, displayH_, renderW_, renderH_)) {
        CRTX_LOG("DLSS : mode %s indisponible à cette résolution", dlssModeName(opt_.dlssMode));
        dlssActive_ = false;
    }
    if (!dlssActive_) {
        renderW_ = displayW_;
        renderH_ = displayH_;
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
            ok = dlss_->createFeature(cmd, opt_.dlssMode, renderW_, renderH_, displayW_, displayH_);
        });
        if (!ok) {
            CRTX_LOG("%s -> rendu natif", dlss_->status().c_str());
            for (SharedBuffer* b : {&shColor_, &shDepth_, &shMotion_, &shDlssOut_}) vk_->destroySharedBuffer(*b);
            for (GpuImage* i : {&imgColor_, &imgDepth_, &imgMotion_, &imgOut_}) vk_->destroyImage(*i);
            dlssActive_ = false;
            renderW_ = displayW_;
            renderH_ = displayH_;
        }
    }
    renderer_->resize(renderW_, renderH_);
    const float ratio = static_cast<float>(displayW_) / static_cast<float>(renderW_);
    jitterPhases_ = std::max(8u, static_cast<unsigned>(std::lround(8.0f * ratio * ratio)));
    resetHistory_ = true;
    hasPrevCamera_ = false;
    if (dlssActive_)
        CRTX_LOG("DLSS %s : rendu %ux%u -> affichage %ux%u (%u phases de jitter Halton)", dlssModeName(opt_.dlssMode),
                 renderW_, renderH_, displayW_, displayH_, jitterPhases_);
    else
        CRTX_LOG("Rendu natif %ux%u (accumulation progressive + débruiteur IA OptiX)", renderW_, renderH_);
}

CameraData App::makeCamera(const Orbit& o, uint32_t w, uint32_t h) const
{
    const float3 target = scene_->focusPoint();
    const float cp = std::cos(o.pitch), sp = std::sin(o.pitch);
    const float3 eye = target + make_float3(cp * std::sin(o.yaw), sp, cp * std::cos(o.yaw)) * o.distance;
    const float3 W = normalize(target - eye);
    const float3 right = normalize(cross(W, make_float3(0, 1, 0)));
    const float3 up = cross(right, W);
    const float tanY = std::tan(o.fovY * 0.5f);
    const float aspect = static_cast<float>(w) / static_cast<float>(std::max(h, 1u));
    CameraData c{};
    c.eye = eye;
    c.U = right * (tanY * aspect);
    c.V = up * tanY;
    c.W = W;
    c.zNear = 0.1f;
    c.zFar = 400.0f;
    return c;
}

void App::cudaSignal(uint64_t value)
{
    cudaExternalSemaphore_t sem = vk_->cudaTimeline();
    cudaExternalSemaphoreSignalParams p{};
    p.params.fence.value = value;
    CUDA_CHECK(cudaSignalExternalSemaphoresAsync(&sem, &p, 1, stream_));
}

void App::cudaWait(uint64_t value)
{
    cudaExternalSemaphore_t sem = vk_->cudaTimeline();
    cudaExternalSemaphoreWaitParams p{};
    p.params.fence.value = value;
    CUDA_CHECK(cudaWaitExternalSemaphoresAsync(&sem, &p, 1, stream_));
}

void App::frame(float dt)
{
    const uint64_t base = frameCount_ * 4;
    vk_->waitTimeline(base);  // l'image précédente est entièrement terminée (CPU <-> GPU)

    // ---- sélection par rayon de l'image précédente -> survol
    hoveredKey_ = rotating_ ? -1 : renderer_->pickResult();

    // ---- état calculatrice -> scène 3D
    scene_->setShift(controller_->shift());
    scene_->setDisplay(controller_->display());
    const bool animated = scene_->animate(dt, hoveredKey_);

    // ---- caméra orbitale lissée
    const float k = std::min(1.0f, dt * 10.0f);
    auto approach = [k](float& v, float target, float eps) {
        v += (target - v) * k;
        if (std::fabs(target - v) < eps) v = target;  // convergence exacte : l'accumulation peut reprendre
    };
    approach(orbit_.yaw, orbitTarget_.yaw, 1e-5f);
    approach(orbit_.pitch, orbitTarget_.pitch, 1e-5f);
    approach(orbit_.distance, orbitTarget_.distance, 1e-4f);
    const CameraData cam = makeCamera(orbit_, renderW_, renderH_);
    if (!hasPrevCamera_) prevCamera_ = cam;
    const bool cameraMoved = length(cam.eye - prevCamera_.eye) > 1e-5f || length(cam.W - prevCamera_.W) > 1e-6f;

    // ---- CUDA : ray tracing + débruitage
    renderer_->syncScene(*scene_, stream_);
    FrameInput in;
    in.camera = cam;
    in.prevCamera = prevCamera_;
    in.dlss = dlssActive_;
    in.frameIndex = static_cast<unsigned>(frameCount_);
    in.time = static_cast<float>(glfwGetTime());
    in.resetAccumulation = cameraMoved || animated || resetHistory_;
    if (dlssActive_) {
        const unsigned idx = static_cast<unsigned>(frameCount_ % jitterPhases_) + 1;
        in.jitter = make_float2(halton(idx, 2) - 0.5f, halton(idx, 3) - 0.5f);
    }
    // Jitter transmis au DLSS (CRTX_DLSS_JITTER_SIGN=-1 inverse la convention, pour diagnostic)
    const float2 dlssJitter = in.jitter * jitterSign_;
    const float4* hdr = renderer_->render(in, settings_, stream_);

    int winW = 1, winH = 1;
    glfwGetWindowSize(window_, &winW, &winH);
    const float sx = static_cast<float>(renderW_) / std::max(winW, 1);
    const float sy = static_cast<float>(renderH_) / std::max(winH, 1);
    renderer_->pick(cam, make_float2(static_cast<float>(cursorX_) * sx, static_cast<float>(cursorY_) * sy), stream_);

    postParams_.frame = static_cast<unsigned>(frameCount_);
    const bool bgra = vk_->swapIsBgra();
    if (dlssActive_) {
        const size_t rn = static_cast<size_t>(renderW_) * renderH_;
        kernels::packHalf4(hdr, static_cast<__half*>(shColor_.cudaPtr), static_cast<int>(rn), stream_);
        CUDA_CHECK(cudaMemcpyAsync(shDepth_.cudaPtr, renderer_->depthBuffer(), rn * sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream_));
        CUDA_CHECK(cudaMemcpyAsync(shMotion_.cudaPtr, renderer_->motionBuffer(), rn * sizeof(float2),
                                   cudaMemcpyDeviceToDevice, stream_));
        cudaSignal(base + 1);

        // ---- Vulkan : DLSS (Tensor cores)
        VK_CHECK(vkResetCommandBuffer(cmdDlss_, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmdDlss_, &bi));
        const std::pair<SharedBuffer*, GpuImage*> inputs[] = {{&shColor_, &imgColor_}, {&shDepth_, &imgDepth_}, {&shMotion_, &imgMotion_}};
        for (const auto& io : inputs) {
            VulkanContext::imageBarrier(cmdDlss_, io.second->image, VK_IMAGE_LAYOUT_UNDEFINED,
                                        VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                                        VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkBufferImageCopy region{};
            region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            region.imageExtent = {renderW_, renderH_, 1};
            vkCmdCopyBufferToImage(cmdDlss_, io.first->buffer, io.second->image, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1,
                                   &region);
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
        if (!dlss_->evaluate(cmdDlss_, images, renderW_, renderH_, displayW_, displayH_, dlssJitter.x, dlssJitter.y,
                             resetHistory_ || !hasPrevCamera_, lastFrameMs_))
            CRTX_LOG("DLSS : échec de l'évaluation");
        VulkanContext::imageBarrier(cmdDlss_, imgOut_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
        VkBufferImageCopy outRegion{};
        outRegion.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        outRegion.imageExtent = {displayW_, displayH_, 1};
        vkCmdCopyImageToBuffer(cmdDlss_, imgOut_.image, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, shDlssOut_.buffer, 1,
                               &outRegion);
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

        cudaWait(base + 2);
        post_->run(shDlssOut_.cudaPtr, true, static_cast<unsigned char*>(shPresent_.cudaPtr), bgra,
                   static_cast<int>(displayW_), static_cast<int>(displayH_), postParams_, stream_);
    } else {
        post_->run(hdr, false, static_cast<unsigned char*>(shPresent_.cudaPtr), bgra, static_cast<int>(displayW_),
                   static_cast<int>(displayH_), postParams_, stream_);
    }
    cudaSignal(base + 3);

    // ---- Vulkan : présentation
    const VkSemaphore tl = vk_->timeline();
    uint32_t imageIndex = 0;
    const VkResult ar = vkAcquireNextImageKHR(vk_->device(), vk_->swapchain(), UINT64_MAX, vk_->imageAvailable(),
                                              VK_NULL_HANDLE, &imageIndex);
    if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
        // termine quand même l'image sur la timeline (valeur base+4)
        const uint64_t waitV = base + 3, sigV = base + 4;
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
        si.signalSemaphoreCount = 1;
        si.pSignalSemaphores = &tl;
        VK_CHECK(vkQueueSubmit(vk_->queue(), 1, &si, VK_NULL_HANDLE));
        swapchainDirty_ = true;
    } else {
        if (ar != VK_SUCCESS && ar != VK_SUBOPTIMAL_KHR) VK_CHECK(ar);
        if (ar == VK_SUBOPTIMAL_KHR) swapchainDirty_ = true;
        const VkImage swapImg = vk_->swapImages()[imageIndex];
        VK_CHECK(vkResetCommandBuffer(cmdPresent_, 0));
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cmdPresent_, &bi));
        VulkanContext::imageBarrier(cmdPresent_, swapImg, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_TRANSFER_WRITE_BIT);
        VkBufferImageCopy region{};
        region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        region.imageExtent = {displayW_, displayH_, 1};
        vkCmdCopyBufferToImage(cmdPresent_, shPresent_.buffer, swapImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
        VulkanContext::imageBarrier(cmdPresent_, swapImg, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                    VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                                    VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
        VK_CHECK(vkEndCommandBuffer(cmdPresent_));

        const VkSemaphore waits[2] = {tl, vk_->imageAvailable()};
        const VkPipelineStageFlags stages[2] = {VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT};
        const uint64_t waitVals[2] = {base + 3, 0};
        const VkSemaphore signals[2] = {vk_->renderFinished(imageIndex), tl};
        const uint64_t sigVals[2] = {0, base + 4};
        VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
        ts.waitSemaphoreValueCount = 2;
        ts.pWaitSemaphoreValues = waitVals;
        ts.signalSemaphoreValueCount = 2;
        ts.pSignalSemaphoreValues = sigVals;
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.pNext = &ts;
        si.waitSemaphoreCount = 2;
        si.pWaitSemaphores = waits;
        si.pWaitDstStageMask = stages;
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmdPresent_;
        si.signalSemaphoreCount = 2;
        si.pSignalSemaphores = signals;
        VK_CHECK(vkQueueSubmit(vk_->queue(), 1, &si, VK_NULL_HANDLE));

        const VkSwapchainKHR swap = vk_->swapchain();
        const VkSemaphore rf = vk_->renderFinished(imageIndex);
        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &rf;
        pi.swapchainCount = 1;
        pi.pSwapchains = &swap;
        pi.pImageIndices = &imageIndex;
        const VkResult pr = vkQueuePresentKHR(vk_->queue(), &pi);
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) swapchainDirty_ = true;
        else if (pr != VK_SUCCESS) VK_CHECK(pr);
    }

    prevCamera_ = cam;
    hasPrevCamera_ = true;
    resetHistory_ = false;
    ++frameCount_;
}

int App::run()
{
    auto last = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        int fbw = 0, fbh = 0;
        glfwGetFramebufferSize(window_, &fbw, &fbh);
        if (fbw == 0 || fbh == 0) {  // fenêtre réduite
            glfwWaitEvents();
            continue;
        }
        if (swapchainDirty_) {
            vk_->waitTimeline(frameCount_ * 4);
            vkDeviceWaitIdle(vk_->device());
            vk_->createSwapchain(static_cast<uint32_t>(fbw), static_cast<uint32_t>(fbh), opt_.vsync);
            configureResolution();
            swapchainDirty_ = false;
        }
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::min(0.1f, std::chrono::duration<float>(now - last).count());
        last = now;
        lastFrameMs_ = dt * 1000.0f;
        frame(dt);
        updateTitle(dt);
    }
    vk_->waitTimeline(frameCount_ * 4);
    return 0;
}

void App::updateTitle(float dt)
{
    titleTimer_ += dt;
    fpsAccum_ += dt;
    ++fpsFrames_;
    if (titleTimer_ < 0.5f) return;
    const float fps = fpsFrames_ / std::max(fpsAccum_, 1e-4f);
    char buf[512];
    if (dlssActive_)
        std::snprintf(buf, sizeof(buf),
                      "CalculatoRTX | %s | DLSS %s %ux%u -> %ux%u | %.0f ips | Path tracing %.2f ms | IA %.2f ms | %u spp x %u rebonds | SER %s",
                      gpuName_.c_str(), dlssModeName(opt_.dlssMode), renderW_, renderH_, displayW_, displayH_, fps,
                      renderer_->lastTraceMs(), renderer_->lastDenoiseMs(), settings_.spp, settings_.maxBounces,
                      settings_.ser ? "on" : "off");
    else
        std::snprintf(buf, sizeof(buf),
                      "CalculatoRTX | %s | Natif %ux%u (DLSS off) | %.0f ips | Path tracing %.2f ms | IA %.2f ms | %u img accumulées | SER %s",
                      gpuName_.c_str(), renderW_, renderH_, fps, renderer_->lastTraceMs(), renderer_->lastDenoiseMs(),
                      renderer_->accumulatedFrames(), settings_.ser ? "on" : "off");
    glfwSetWindowTitle(window_, buf);
    titleTimer_ = 0.0f;
    fpsAccum_ = 0.0f;
    fpsFrames_ = 0;
}

// ============================================================================ entrées
void App::pressKey(KeyId id)
{
    const auto& keys = scene_->keys();
    for (size_t i = 0; i < keys.size(); ++i) {
        if (keys[i].id == id) {
            scene_->pressKey(static_cast<int>(i));
            break;
        }
    }
    controller_->press(id);
}

void App::onKeyChar(unsigned int c)
{
    KeyId id = KeyId::Count;
    switch (c) {
        case '0': id = KeyId::D0; break;
        case '1': id = KeyId::D1; break;
        case '2': id = KeyId::D2; break;
        case '3': id = KeyId::D3; break;
        case '4': id = KeyId::D4; break;
        case '5': id = KeyId::D5; break;
        case '6': id = KeyId::D6; break;
        case '7': id = KeyId::D7; break;
        case '8': id = KeyId::D8; break;
        case '9': id = KeyId::D9; break;
        case '.': case ',': id = KeyId::Dot; break;
        case '+': id = KeyId::Add; break;
        case '-': id = KeyId::Sub; break;
        case '*': case 'x': case 'X': id = KeyId::Mul; break;
        case '/': id = KeyId::Div; break;
        case '^': id = KeyId::Pow; break;
        case '(': id = KeyId::LParen; break;
        case ')': id = KeyId::RParen; break;
        case '!': id = KeyId::Fact; break;
        case '=': id = KeyId::Equals; break;
        case 'p': case 'P': id = KeyId::Pi; break;
        case 'e': id = KeyId::Euler; break;
        case 'E': id = KeyId::Exp; break;
        case 's': case 'S': id = KeyId::Sin; break;
        case 'c': case 'C': id = KeyId::Cos; break;
        case 't': case 'T': id = KeyId::Tan; break;
        case 'l': case 'L': id = KeyId::Ln; break;
        case 'g': case 'G': id = KeyId::Log; break;
        case 'r': case 'R': id = KeyId::Sqrt; break;
        case 'q': case 'Q': id = KeyId::Square; break;
        case 'a': case 'A': id = KeyId::Ans; break;
        case 'n': case 'N': id = KeyId::Negate; break;
        case 'i': case 'I': id = KeyId::Inverse; break;
        case 'd': case 'D': id = KeyId::Drg; break;
        case 'm': id = KeyId::MemRecall; break;
        case 'M': id = KeyId::MemPlus; break;
        default: break;
    }
    if (id != KeyId::Count) pressKey(id);
}

void App::onKey(int key, int action, int)
{
    if (action != GLFW_PRESS && action != GLFW_REPEAT) return;
    switch (key) {
        case GLFW_KEY_ENTER: case GLFW_KEY_KP_ENTER: pressKey(KeyId::Equals); break;
        case GLFW_KEY_BACKSPACE: pressKey(KeyId::Backspace); break;
        case GLFW_KEY_ESCAPE: case GLFW_KEY_DELETE: pressKey(KeyId::AllClear); break;
        case GLFW_KEY_TAB: pressKey(KeyId::Second); break;
        case GLFW_KEY_F1: printHelp(); break;
        case GLFW_KEY_F2:
            opt_.dlss = !opt_.dlss;
            swapchainDirty_ = true;
            CRTX_LOG("DLSS %s", opt_.dlss ? "activé" : "désactivé");
            break;
        case GLFW_KEY_F3: {
            const DlssMode order[] = {DlssMode::Dlaa, DlssMode::Quality, DlssMode::Balanced, DlssMode::Performance,
                                      DlssMode::UltraPerformance};
            int i = 0;
            while (order[i] != opt_.dlssMode) ++i;
            opt_.dlssMode = order[(i + 1) % 5];
            swapchainDirty_ = true;
            CRTX_LOG("Mode DLSS : %s", dlssModeName(opt_.dlssMode));
            break;
        }
        case GLFW_KEY_F4: settings_.denoise = !settings_.denoise; CRTX_LOG("Débruiteur IA : %s", settings_.denoise ? "on" : "off"); break;
        case GLFW_KEY_F5: settings_.ser = !settings_.ser; CRTX_LOG("Shader Execution Reordering : %s", settings_.ser ? "on" : "off"); break;
        case GLFW_KEY_F6: opt_.vsync = !opt_.vsync; swapchainDirty_ = true; break;
        case GLFW_KEY_F7: {
            const unsigned b[] = {2, 3, 5, 8};
            int i = 0;
            while (i < 3 && b[i] != settings_.maxBounces) ++i;
            settings_.maxBounces = b[(i + 1) % 4];
            resetHistory_ = true;
            CRTX_LOG("Rebonds max : %u", settings_.maxBounces);
            break;
        }
        case GLFW_KEY_F8:
            settings_.spp = settings_.spp >= 4 ? 1 : settings_.spp * 2;
            resetHistory_ = true;
            CRTX_LOG("Échantillons par pixel : %u", settings_.spp);
            break;
        case GLFW_KEY_F9: postParams_.exposure *= 0.8f; break;
        case GLFW_KEY_F10: postParams_.exposure *= 1.25f; break;
        case GLFW_KEY_HOME: orbitTarget_ = Orbit{}; break;
        default: break;
    }
}

void App::onMouseButton(int button, int action)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        const int k = renderer_ ? renderer_->pickResult() : -1;
        if (k >= 0 && k < static_cast<int>(scene_->keys().size())) pressKey(scene_->keys()[k].id);
        else {
            rotating_ = true;  // clic hors des touches : rotation de la vue
            dragX_ = cursorX_;
            dragY_ = cursorY_;
        }
    }
    if ((button == GLFW_MOUSE_BUTTON_RIGHT || button == GLFW_MOUSE_BUTTON_MIDDLE) && action == GLFW_PRESS) {
        rotating_ = true;
        dragX_ = cursorX_;
        dragY_ = cursorY_;
    }
    if (action == GLFW_RELEASE) rotating_ = false;
}

void App::onCursor(double x, double y)
{
    if (rotating_) {
        orbitTarget_.yaw -= static_cast<float>(x - dragX_) * 0.005f;
        orbitTarget_.pitch += static_cast<float>(y - dragY_) * 0.004f;
        orbitTarget_.yaw = std::clamp(orbitTarget_.yaw, -0.9f, 0.9f);
        orbitTarget_.pitch = std::clamp(orbitTarget_.pitch, 0.35f, 1.45f);
        dragX_ = x;
        dragY_ = y;
    }
    cursorX_ = x;
    cursorY_ = y;
}

void App::onScroll(double dy)
{
    orbitTarget_.distance = std::clamp(orbitTarget_.distance * static_cast<float>(std::pow(0.9, dy)), 16.0f, 60.0f);
}

void App::printHelp() const
{
    std::printf(
        "\n=== CalculatoRTX : commandes ===\n"
        "  Souris gauche : appuyer sur une touche (sélection par lancer de rayon)\n"
        "  Glisser (gauche hors touche / droite / milieu) : tourner la vue ; molette : zoom ; Origine : recentrer\n"
        "  Clavier : 0-9 . + - * / ^ ( ) ! =/Entrée, Retour arrière, Échap (AC), Tab (2nd)\n"
        "            s c t (sin cos tan)  l (ln)  g (log)  r (racine)  q (x^2)  i (1/x)  p (pi)  e  E (EXP)\n"
        "            a (ANS)  n (+/-)  d (DEG/RAD)  m (MR)  M (M+)\n"
        "  F2 DLSS on/off | F3 mode DLSS | F4 débruiteur IA | F5 SER | F6 V-Sync | F7 rebonds | F8 spp\n"
        "  F9/F10 exposition -/+ | F1 cette aide\n\n");
    std::fflush(stdout);
}

}  // namespace crtx
