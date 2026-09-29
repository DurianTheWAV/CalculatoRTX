// CalculatoRTX - calculatrice scientifique entièrement ray tracée (OptiX / RT cores),
// super-résolution DLSS (Tensor cores) et calculs exécutés sur le GPU (CUDA).
#include "app/App.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>

#if defined(_WIN32)
#include <clocale>
#endif

namespace {

void usage()
{
    std::printf(
        "Usage : CalculatoRTX [options]\n"
        "  --no-dlss                 rendu natif (accumulation + débruiteur IA)\n"
        "  --dlss-mode=<mode>        dlaa | quality | balanced | performance | ultra\n"
        "  --size=<L>x<H>            taille de fenêtre (défaut 1600x1000)\n"
        "  --no-vsync                présentation sans synchronisation verticale\n"
        "  --device=<n>              index du GPU CUDA\n"
        "  --validation              active les couches de validation Vulkan\n");
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
        if (a == "--no-dlss") opt.dlss = false;
        else if (a == "--no-vsync") opt.vsync = false;
        else if (a == "--validation") opt.validation = true;
        else if (a.rfind("--device=", 0) == 0) opt.cudaDevice = std::atoi(a.c_str() + 9);
        else if (a.rfind("--size=", 0) == 0) {
            int w = 0, h = 0;
            if (std::sscanf(a.c_str() + 7, "%dx%d", &w, &h) == 2 && w > 0 && h > 0) {
                opt.width = w;
                opt.height = h;
            }
        } else if (a.rfind("--dlss-mode=", 0) == 0) {
            const std::string m = a.substr(12);
            if (m == "dlaa") opt.dlssMode = crtx::DlssMode::Dlaa;
            else if (m == "quality") opt.dlssMode = crtx::DlssMode::Quality;
            else if (m == "balanced") opt.dlssMode = crtx::DlssMode::Balanced;
            else if (m == "performance") opt.dlssMode = crtx::DlssMode::Performance;
            else if (m == "ultra") opt.dlssMode = crtx::DlssMode::UltraPerformance;
        } else if (a == "-h" || a == "--help") {
            usage();
            return 0;
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
