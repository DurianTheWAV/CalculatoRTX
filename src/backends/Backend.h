// CalculatoRTX - interface commune des deux moteurs de rendu + calcul.
//
//   * NvidiaBackend : CUDA + OptiX (RT cores, SER, Opacity Micromaps) + DLSS + débruiteur IA
//   * VulkanBackend : ray tracing Vulkan (VK_KHR_ray_query) + débruiteur à-trous + AMD FSR 1,
//                     calcul FP64 en compute shader - AMD Radeon RX 6000+, Intel Arc, NVIDIA RTX
//
// Stratégie de rendu commune (optimisée pour les GPU milieu de gamme type RTX 4060 Ti) :
//   * en mouvement (caméra, touche qui s'enfonce, frappe) : résolution réduite + DLSS / FSR ;
//   * immobile : accumulation progressive en résolution native, affichée dès qu'elle
//     dépasse l'image en mouvement, puis arrêt complet du ray tracing une fois l'image
//     convergée (le GPU repasse au repos).
#pragma once

#include "../calc/ICalcEngine.h"
#include "../render/SceneTypes.h"
#include "../scene/CalculatorScene.h"

#include <cstdint>
#include <string>

struct GLFWwindow;

namespace crtx {

enum class UpscaleMode { Auto, Native, Quality, Balanced, Performance, UltraPerformance };
const char* upscaleModeName(UpscaleMode m);

struct BackendOptions {
    bool validation = false;
    bool vsync = true;
    bool upscale = true;  // DLSS (NVIDIA) ou FSR 1 (Vulkan) pendant les mouvements
    UpscaleMode mode = UpscaleMode::Auto;
    int device = -1;  // index du GPU (CUDA ou Vulkan), -1 = automatique
};

struct RenderSettings {
    unsigned spp = 1;           // échantillons par pixel et par image en mouvement
    unsigned maxBounces = 5;    // rebonds diffus / spéculaires (hors traversées de verre)
    bool ser = true;            // Shader Execution Reordering (NVIDIA)
    bool denoise = true;        // débruiteur (IA OptiX ou à-trous)
    bool fireflyClamp = true;
    unsigned convergedSamples = 1024;  // arrêt du ray tracing une fois atteint (image immobile)
};

struct PostSettings {
    float exposure = 1.0f;
    float bloomStrength = 0.08f;  // part de l'énergie au-dessus du seuil étalée en halo
    float bloomThreshold = 1.2f;
    float vignette = 0.22f;
};

struct FrameContext {
    CalculatorScene* scene = nullptr;
    CameraData camera{};
    CameraData prevCamera{};
    bool changed = true;      // caméra, animation ou contenu modifiés depuis l'image précédente
    uint64_t frameIndex = 0;
    float dt = 0.016f;
    float time = 0.0f;
    float cursorU = 0.0f, cursorV = 0.0f;  // curseur en coordonnées normalisées [0,1] de la fenêtre
    bool hoverEnabled = true;
    RenderSettings settings;
    PostSettings post;
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual std::string name() const = 0;          // ex. "NVIDIA CUDA/OptiX", "Vulkan"
    virtual std::string deviceName() const = 0;
    virtual calc::ICalcEngine& calc() = 0;

    // (Re)crée la swapchain et les tampons dépendant de la résolution.
    virtual void resize(uint32_t framebufferW, uint32_t framebufferH) = 0;
    // Rend (si nécessaire) et présente une image. false si la swapchain doit être recréée.
    virtual bool frame(const FrameContext& ctx) = 0;
    // Touche sous le curseur (lancer de rayon de l'image précédente), -1 si aucune.
    virtual int hoveredKey() const = 0;
    // Vrai quand l'image immobile est convergée : la boucle peut attendre les évènements.
    virtual bool idle() const = 0;
    // Aspect (largeur / hauteur) de la zone rendue.
    virtual float aspect() const = 0;

    virtual void setVsync(bool vsync) = 0;
    virtual void setUpscale(bool enabled, UpscaleMode mode) = 0;
    virtual bool upscaleEnabled() const = 0;
    virtual UpscaleMode upscaleMode() const = 0;
    virtual std::string upscalerName() const = 0;  // "DLSS", "FSR 1"

    virtual std::string stats(float fps) const = 0;  // titre de la fenêtre
    virtual bool saveScreenshot(const std::string& path) = 0;
    virtual void waitIdle() = 0;
};

}  // namespace crtx
