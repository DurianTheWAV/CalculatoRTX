// CalculatoRTX - calculatrice scientifique entièrement ray tracée, calculs exécutés sur le GPU.
//   * NVIDIA : OptiX (RT cores) + DLSS (Tensor cores) + CUDA
//   * AMD / Intel / NVIDIA : ray tracing Vulkan + AMD FSR 1 + compute shaders FP64
#include "app/App.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#if defined(_WIN32)
#include <clocale>
#include <windows.h>
#endif

namespace {

void usage()
{
    std::printf(
        "Usage : CalculatoRTX [options]\n"
        "  --backend=<b>             auto | nvidia (CUDA + OptiX + DLSS) | vulkan (AMD / Intel / NVIDIA)\n"
        "  --no-upscale              pas de DLSS / FSR en mouvement (toujours en résolution native)\n"
        "  --upscale-mode=<mode>     auto | native | quality | balanced | performance | ultra\n"
        "  --no-dlss, --dlss-mode=   anciens noms de --no-upscale / --upscale-mode\n"
        "  --size=<L>x<H>            taille de fenêtre (défaut 1600x1000)\n"
        "  --no-vsync                présentation sans synchronisation verticale\n"
        "  --device=<n>              index du GPU (CUDA ou Vulkan)\n"
        "  --validation              active les couches de validation Vulkan\n"
        "  --type=<touches>          touches tapées au démarrage (ex. \"s30+2^10*3=\")\n"
        "  --screenshot=<f.bmp>      capture après --frames images, puis fermeture\n"
        "  --frames=<n>              nombre d'images avant fermeture (avec --screenshot)\n");
}

bool parseMode(const std::string& m, crtx::UpscaleMode& out)
{
    using crtx::UpscaleMode;
    if (m == "auto") out = UpscaleMode::Auto;
    else if (m == "native" || m == "dlaa") out = UpscaleMode::Native;
    else if (m == "quality") out = UpscaleMode::Quality;
    else if (m == "balanced") out = UpscaleMode::Balanced;
    else if (m == "performance") out = UpscaleMode::Performance;
    else if (m == "ultra") out = UpscaleMode::UltraPerformance;
    else return false;
    return true;
}

}  // namespace

int main(int argc, char** argv)
{
#if defined(_WIN32)
    std::setlocale(LC_ALL, ".UTF8");
    SetConsoleOutputCP(CP_UTF8);
#endif
    crtx::AppOptions opt;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto value = [&](const char* prefix) { return a.substr(std::strlen(prefix)); };
        if (a == "--no-dlss" || a == "--no-upscale") opt.backendOptions.upscale = false;
        else if (a == "--no-vsync") opt.backendOptions.vsync = false;
        else if (a == "--validation") opt.backendOptions.validation = true;
        else if (a.rfind("--device=", 0) == 0) opt.backendOptions.device = std::atoi(value("--device=").c_str());
        else if (a.rfind("--backend=", 0) == 0) {
            const std::string b = value("--backend=");
            if (b == "nvidia" || b == "cuda" || b == "optix") opt.backend = crtx::BackendChoice::Nvidia;
            else if (b == "vulkan" || b == "amd" || b == "vk") opt.backend = crtx::BackendChoice::Vulkan;
            else opt.backend = crtx::BackendChoice::Auto;
        } else if (a.rfind("--size=", 0) == 0) {
            int w = 0, h = 0;
            if (std::sscanf(a.c_str() + 7, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                opt.width = w;
                opt.height = h;
            }
        } else if (a.rfind("--upscale-mode=", 0) == 0) {
            if (!parseMode(value("--upscale-mode="), opt.backendOptions.mode)) std::fprintf(stderr, "Mode inconnu : %s\n", a.c_str());
        } else if (a.rfind("--dlss-mode=", 0) == 0) {
            if (!parseMode(value("--dlss-mode="), opt.backendOptions.mode)) std::fprintf(stderr, "Mode inconnu : %s\n", a.c_str());
        } else if (a.rfind("--type=", 0) == 0) {
            opt.type = value("--type=");
        } else if (a.rfind("--screenshot=", 0) == 0) {
            opt.screenshotPath = value("--screenshot=");
            if (opt.frames == 0) opt.frames = 120;
        } else if (a.rfind("--frames=", 0) == 0) {
            opt.frames = std::atoi(value("--frames=").c_str());
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
        } else {
            std::fprintf(stderr, "Option inconnue : %s (--help)\n", a.c_str());
        }
    }
    try {
        crtx::App app(opt);
        return app.run();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "\n[CalculatoRTX] ERREUR FATALE : %s\n", e.what());
        return EXIT_FAILURE;
    }
}
