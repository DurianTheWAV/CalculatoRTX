// CalculatoRTX - boucle principale, entrées et choix du backend.
#include "App.h"

#include "../backends/VulkanBackend.h"
#if defined(CRTX_WITH_NVIDIA_BACKEND)
#include "../backends/NvidiaBackend.h"
#endif
#include "../gpu/VulkanCommon.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>

namespace crtx {

namespace {

App* appFrom(GLFWwindow* w) { return static_cast<App*>(glfwGetWindowUserPointer(w)); }

// Mention du bandeau : nom court du GPU réellement utilisé + technologies du backend
std::string makeBadge(const Backend& b)
{
    std::string dev = b.deviceName();
    for (const char* prefix : {"NVIDIA GeForce ", "NVIDIA ", "AMD ", "Intel(R) ", "Intel "})
        if (dev.rfind(prefix, 0) == 0) {
            dev = dev.substr(std::strlen(prefix));
            break;
        }
    const size_t paren = dev.find(" (");
    if (paren != std::string::npos) dev.resize(paren);
    std::string clean;
    for (char c : dev)
        if (c >= 32 && c < 127 && c != '|') clean += c;
    const bool nvidia = b.name().find("NVIDIA") != std::string::npos;
    return clean + (nvidia ? "|OptiX|DLSS|CUDA" : "|Vulkan RT|FSR 1");
}

}  // namespace

App::App(const AppOptions& opt) : opt_(opt)
{
    initWindow();
    createBackend();
    if (calc::runSelfTest(backend_->calc()))
        CRTX_LOG("Moteur de calcul %s : auto-test OK (double-double / FP64 / FP32 / intervalles)",
                 backend_->calc().apiName());
    controller_ = std::make_unique<CalculatorController>(backend_->calc());
    scene_ = std::make_unique<CalculatorScene>(font_);
    scene_->setBadge(makeBadge(*backend_));
    scene_->build();
    orbitTarget_ = orbit_;
    for (char c : opt_.type) {
        if (c == '\n') pressKey(KeyId::Equals);
        else onKeyChar(static_cast<unsigned char>(c));
    }
    printHelp();
}

App::~App()
{
    if (backend_) backend_->waitIdle();
    controller_.reset();
    scene_.reset();
    backend_.reset();
    if (window_) glfwDestroyWindow(window_);
    glfwTerminate();
}

void App::initWindow()
{
    if (!glfwInit()) throwError("Échec d'initialisation de GLFW", __FILE__, __LINE__);
    if (!glfwVulkanSupported()) throwError("Vulkan indisponible (pilote graphique / chargeur Vulkan ?)", __FILE__, __LINE__);
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

// Choix du backend : NVIDIA (CUDA + OptiX + DLSS) si compilé et utilisable, sinon Vulkan.
void App::createBackend()
{
#if defined(CRTX_WITH_NVIDIA_BACKEND)
    if (opt_.backend != BackendChoice::Vulkan) {
        try {
            backend_ = createNvidiaBackend(window_, opt_.backendOptions);
        } catch (const std::exception& e) {
            if (opt_.backend == BackendChoice::Nvidia) throw;
            CRTX_LOG("Backend NVIDIA indisponible (%s) -> backend Vulkan", e.what());
            backend_.reset();  // sa surface Vulkan éventuelle est détruite : la fenêtre est réutilisable
        }
    }
#else
    if (opt_.backend == BackendChoice::Nvidia)
        throwError("Ce binaire a été compilé sans CUDA : backend NVIDIA indisponible (utilisez --backend=vulkan)",
                   __FILE__, __LINE__);
#endif
    if (!backend_) backend_ = createVulkanBackend(window_, opt_.backendOptions);
    CRTX_LOG("Backend de rendu : %s sur %s", backend_->name().c_str(), backend_->deviceName().c_str());
}

CameraData App::makeCamera(const Orbit& o) const
{
    const float3 target = scene_->focusPoint();
    const float cp = std::cos(o.pitch), sp = std::sin(o.pitch);
    const float3 eye = target + make_float3(cp * std::sin(o.yaw), sp, cp * std::cos(o.yaw)) * o.distance;
    const float3 W = normalize(target - eye);
    const float3 right = normalize(cross(W, make_float3(0, 1, 0)));
    const float3 up = cross(right, W);
    const float tanY = std::tan(o.fovY * 0.5f);
    const float aspect = backend_->aspect();
    CameraData c{};
    c.eye = eye;
    c.U = right * (tanY * aspect);
    c.V = up * tanY;
    c.W = W;
    c.zNear = 0.1f;
    c.zFar = 400.0f;
    return c;
}

void App::frame(float dt)
{
    // ---- survol : résultat du rayon de sélection de l'image précédente
    hoveredKey_ = rotating_ ? -1 : backend_->hoveredKey();

    // ---- état calculatrice -> scène 3D
    scene_->setShift(controller_->shift());
    scene_->setDisplay(controller_->display());
    const bool animated = scene_->animate(dt, hoveredKey_);
    bool geometryChanged = false;
    for (const SceneMesh& m : scene_->meshes()) geometryChanged |= m.dirty;

    // ---- caméra orbitale lissée
    const float k = std::min(1.0f, dt * 10.0f);
    auto approach = [k](float& v, float target, float eps) {
        v += (target - v) * k;
        if (std::fabs(target - v) < eps) v = target;  // convergence exacte : l'accumulation peut reprendre
    };
    approach(orbit_.yaw, orbitTarget_.yaw, 1e-5f);
    approach(orbit_.pitch, orbitTarget_.pitch, 1e-5f);
    approach(orbit_.distance, orbitTarget_.distance, 1e-4f);
    const CameraData cam = makeCamera(orbit_);
    if (!hasPrevCamera_) prevCamera_ = cam;
    const bool cameraMoved = length(cam.eye - prevCamera_.eye) > 1e-5f || length(cam.W - prevCamera_.W) > 1e-6f ||
                             length(cam.U - prevCamera_.U) > 1e-6f;

    FrameContext ctx;
    ctx.scene = scene_.get();
    ctx.camera = cam;
    ctx.prevCamera = prevCamera_;
    ctx.changed = cameraMoved || animated || geometryChanged || resetHistory_;
    ctx.frameIndex = frameCount_;
    ctx.dt = dt;
    ctx.time = static_cast<float>(glfwGetTime());
    int winW = 1, winH = 1;
    glfwGetWindowSize(window_, &winW, &winH);
    ctx.cursorU = static_cast<float>(cursorX_ / std::max(winW, 1));
    ctx.cursorV = static_cast<float>(cursorY_ / std::max(winH, 1));
    ctx.hoverEnabled = !rotating_;
    ctx.settings = settings_;
    ctx.post = post_;
    if (!backend_->frame(ctx)) swapchainDirty_ = true;

    prevCamera_ = cam;
    hasPrevCamera_ = true;
    resetHistory_ = false;
    ++frameCount_;
}

int App::run()
{
    auto last = std::chrono::steady_clock::now();
    while (!glfwWindowShouldClose(window_)) {
        // Image convergée : le GPU est au repos, on attend les évènements (souris, clavier)
        if (backend_->idle() && !swapchainDirty_ && !rotating_) glfwWaitEventsTimeout(0.25);
        else glfwPollEvents();
        int fbw = 0, fbh = 0;
        glfwGetFramebufferSize(window_, &fbw, &fbh);
        if (fbw == 0 || fbh == 0) {  // fenêtre réduite
            glfwWaitEvents();
            continue;
        }
        if (swapchainDirty_) {
            backend_->resize(static_cast<uint32_t>(fbw), static_cast<uint32_t>(fbh));
            swapchainDirty_ = false;
            resetHistory_ = true;
            hasPrevCamera_ = false;
        }
        const auto now = std::chrono::steady_clock::now();
        const float dt = std::min(0.1f, std::chrono::duration<float>(now - last).count());
        last = now;
        frame(dt);
        updateTitle(dt);
        if (opt_.frames > 0 && static_cast<int>(frameCount_) >= opt_.frames) {
            if (!opt_.screenshotPath.empty()) screenshot(opt_.screenshotPath);
            break;
        }
    }
    backend_->waitIdle();
    return 0;
}

void App::updateTitle(float dt)
{
    titleTimer_ += dt;
    fpsAccum_ += dt;
    ++fpsFrames_;
    if (titleTimer_ < 0.5f) return;
    fps_ = fpsFrames_ / std::max(fpsAccum_, 1e-4f);
    glfwSetWindowTitle(window_, backend_->stats(fps_).c_str());
    titleTimer_ = 0.0f;
    fpsAccum_ = 0.0f;
    fpsFrames_ = 0;
}

void App::screenshot(const std::string& path)
{
    if (backend_->saveScreenshot(path)) CRTX_LOG("Capture enregistrée : %s", path.c_str());
    else CRTX_LOG("Capture impossible : %s", path.c_str());
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
            backend_->setUpscale(!backend_->upscaleEnabled(), backend_->upscaleMode());
            swapchainDirty_ = true;
            CRTX_LOG("%s en mouvement : %s", backend_->upscalerName().c_str(), backend_->upscaleEnabled() ? "activé" : "désactivé");
            break;
        case GLFW_KEY_F3: {
            const UpscaleMode order[] = {UpscaleMode::Auto, UpscaleMode::Native, UpscaleMode::Quality, UpscaleMode::Balanced,
                                         UpscaleMode::Performance, UpscaleMode::UltraPerformance};
            int i = 0;
            while (i < 5 && order[i] != backend_->upscaleMode()) ++i;
            const UpscaleMode next = order[(i + 1) % 6];
            backend_->setUpscale(backend_->upscaleEnabled(), next);
            swapchainDirty_ = true;
            CRTX_LOG("Mode %s : %s", backend_->upscalerName().c_str(), upscaleModeName(next));
            break;
        }
        case GLFW_KEY_F4:
            settings_.denoise = !settings_.denoise;
            resetHistory_ = true;
            CRTX_LOG("Débruiteur : %s", settings_.denoise ? "on" : "off");
            break;
        case GLFW_KEY_F5:
            settings_.ser = !settings_.ser;
            resetHistory_ = true;
            CRTX_LOG("Shader Execution Reordering (NVIDIA) : %s", settings_.ser ? "on" : "off");
            break;
        case GLFW_KEY_F6: {
            static bool vsync = true;
            vsync = !vsync;
            backend_->setVsync(vsync);
            swapchainDirty_ = true;
            CRTX_LOG("V-Sync : %s", vsync ? "on" : "off");
            break;
        }
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
            CRTX_LOG("Échantillons par pixel en mouvement : %u", settings_.spp);
            break;
        case GLFW_KEY_F9: post_.exposure *= 0.8f; resetHistory_ = true; break;
        case GLFW_KEY_F10: post_.exposure *= 1.25f; resetHistory_ = true; break;
        case GLFW_KEY_F12: {
            char name[64];
            std::snprintf(name, sizeof(name), "CalculatoRTX_capture_%03d.bmp", screenshotIndex_++);
            screenshot(name);
            break;
        }
        case GLFW_KEY_HOME: orbitTarget_ = Orbit{}; break;
        default: break;
    }
}

void App::onMouseButton(int button, int action)
{
    if (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS) {
        const int k = backend_->hoveredKey();
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
        "  F2 %s on/off | F3 mode %s | F4 débruiteur | F5 SER | F6 V-Sync | F7 rebonds | F8 spp\n"
        "  F9/F10 exposition -/+ | F12 capture d'écran (BMP) | F1 cette aide\n\n",
        backend_->upscalerName().c_str(), backend_->upscalerName().c_str());
    std::fflush(stdout);
}

}  // namespace crtx
