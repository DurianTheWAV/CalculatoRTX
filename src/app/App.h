// CalculatoRTX - application : fenêtre, entrées, caméra, choix du backend de rendu.
#pragma once

#include "../backends/Backend.h"
#include "../scene/CalculatorScene.h"
#include "../scene/StrokeFont.h"
#include "CalculatorController.h"

#include <memory>
#include <string>

struct GLFWwindow;

namespace crtx {

enum class BackendChoice { Auto, Nvidia, Vulkan };

struct AppOptions {
    int width = 1600;
    int height = 1000;
    BackendChoice backend = BackendChoice::Auto;
    BackendOptions backendOptions;
    std::string type;            // touches tapées au démarrage (démonstration, captures)
    std::string screenshotPath;  // capture BMP puis fermeture après 'frames' images (0 = interactif)
    int frames = 0;
};

class App {
public:
    explicit App(const AppOptions& opt);
    ~App();
    int run();

private:
    struct Orbit {
        float yaw = 0.0f;
        float pitch = 0.90f;  // ~52° (la calculatrice est elle-même inclinée de 15°)
        float distance = 32.0f;
        float fovY = 0.54f;   // ~31°
    };

    void initWindow();
    void createBackend();
    CameraData makeCamera(const Orbit& o) const;
    void frame(float dt);
    void onKeyChar(unsigned int codepoint);
    void onKey(int key, int action, int mods);
    void onMouseButton(int button, int action);
    void onCursor(double x, double y);
    void onScroll(double dy);
    void pressKey(KeyId id);
    void printHelp() const;
    void updateTitle(float dt);
    void screenshot(const std::string& path);

    AppOptions opt_;
    GLFWwindow* window_ = nullptr;
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<CalculatorController> controller_;
    StrokeFont font_;
    std::unique_ptr<CalculatorScene> scene_;

    RenderSettings settings_;
    PostSettings post_;
    uint64_t frameCount_ = 0;
    Orbit orbit_, orbitTarget_;
    CameraData prevCamera_{};
    bool hasPrevCamera_ = false;
    bool swapchainDirty_ = true;
    bool resetHistory_ = true;
    int hoveredKey_ = -1;

    double cursorX_ = 0, cursorY_ = 0;
    bool rotating_ = false;
    double dragX_ = 0, dragY_ = 0;

    float titleTimer_ = 0.0f;
    float fpsAccum_ = 0.0f;
    int fpsFrames_ = 0;
    float fps_ = 0.0f;
    int screenshotIndex_ = 0;
};

}  // namespace crtx
