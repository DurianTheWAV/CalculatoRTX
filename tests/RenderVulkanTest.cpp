// CalculatoRTX - rendu hors écran par le backend Vulkan (ray query), sans fenêtre.
//
// Produit une image BMP de la calculatrice et vérifie qu'elle n'est ni noire, ni saturée,
// ni invalide. Sans GPU Vulkan à ray tracing, le test est ignoré (code 77). Fonctionne avec
// le pilote logiciel Mesa lavapipe (lent) : VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json
//   render_vulkan_test --size=800x500 --frames=64 --out=image.bmp [--motion]
#include "../src/backends/VulkanBackend.h"
#include "../src/common/Glyphs.h"
#include "../src/scene/StrokeFont.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

using namespace crtx;

namespace {

CameraData makeCamera(const CalculatorScene& scene, float aspect, float yaw, float pitch, float dist)
{
    const float fovY = 0.54f;
    const float3 target = scene.focusPoint();
    const float cp = std::cos(pitch), sp = std::sin(pitch);
    const float3 eye = target + make_float3(cp * std::sin(yaw), sp, cp * std::cos(yaw)) * dist;
    const float3 W = normalize(target - eye);
    const float3 right = normalize(cross(W, make_float3(0, 1, 0)));
    const float3 up = cross(right, W);
    const float tanY = std::tan(fovY * 0.5f);
    CameraData c{};
    c.eye = eye;
    c.U = right * (tanY * aspect);
    c.V = up * tanY;
    c.W = W;
    c.zNear = 0.1f;
    c.zFar = 400.0f;
    return c;
}

}  // namespace

int main(int argc, char** argv)
{
    uint32_t w = 160, h = 100;
    int frames = 2;
    std::string out = "render_vulkan_test.bmp";
    bool motion = false;
    int orbit = 0;  // dernières images en rotation lente (accumulation temporelle + FSR)
    std::string badge, status = "VK 40us  DD";
    float yaw = 0.0f, pitch = 0.90f, dist = 32.0f;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a.rfind("--size=", 0) == 0) std::sscanf(a.c_str() + 7, "%ux%u", &w, &h);
        else if (a.rfind("--frames=", 0) == 0) frames = std::atoi(a.c_str() + 9);
        else if (a.rfind("--out=", 0) == 0) out = a.substr(6);
        else if (a == "--motion") motion = true;  // dernière image en mouvement (FSR 1)
        else if (a.rfind("--orbit=", 0) == 0) orbit = std::atoi(a.c_str() + 8);
        else if (a.rfind("--badge=", 0) == 0) badge = a.substr(8);    // mention du bandeau ('|' = séparateur)
        else if (a.rfind("--status=", 0) == 0) status = a.substr(9);  // état affiché en haut à droite
        else if (a.rfind("--view=", 0) == 0) std::sscanf(a.c_str() + 7, "%f,%f,%f", &yaw, &pitch, &dist);
    }
    std::unique_ptr<Backend> backend;
    try {
        BackendOptions opt;
        backend = createVulkanBackend(nullptr, opt);
    } catch (const std::exception& e) {
        std::printf("IGNORÉ : pas de GPU Vulkan à ray tracing (%s)\n", e.what());
        return 77;
    }
    try {
        StrokeFont font;
        CalculatorScene scene(font);
        if (!badge.empty()) scene.setBadge(badge);
        scene.build();
        DisplayContent dc;
        dc.expression = std::string("sin(30)+2^10") + char(glyph::kTimes) + "3=";
        dc.main = "3072.5";
        dc.statusLeft = "DEG   M";
        dc.statusRight = status + " " + char(glyph::kCheck);
        scene.setDisplay(dc);
        backend->resize(w, h);
        CameraData prev = makeCamera(scene, backend->aspect(), yaw, pitch, dist);
        for (int i = 0; i < frames; ++i) {
            scene.animate(0.016f, -1);
            const bool turning = i >= frames - orbit;
            const float y = yaw + (turning ? 0.004f * static_cast<float>(i - (frames - orbit) + 1) : 0.0f);
            const CameraData cam = makeCamera(scene, backend->aspect(), y, pitch, dist);
            FrameContext ctx;
            ctx.scene = &scene;
            ctx.camera = cam;
            ctx.prevCamera = prev;
            prev = cam;
            ctx.changed = i == 0 || turning || (motion && i == frames - 1);
            ctx.frameIndex = static_cast<uint64_t>(i);
            ctx.cursorU = 0.62f;  // au-dessus d'une touche
            ctx.cursorV = 0.55f;
            backend->frame(ctx);
        }
        if (!backend->saveScreenshot(out)) {
            std::printf("ÉCHEC : capture impossible\n");
            return 1;
        }
        // Contrôle de l'image : luminance moyenne raisonnable, pas d'image uniforme
        std::ifstream f(out, std::ios::binary);
        std::vector<unsigned char> bmp((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
        if (bmp.size() < 54) return 1;
        double sum = 0.0, sum2 = 0.0;
        size_t n = 0;
        for (size_t i = 54; i + 2 < bmp.size(); i += 3, ++n) {
            const double l = (0.0722 * bmp[i] + 0.7152 * bmp[i + 1] + 0.2126 * bmp[i + 2]) / 255.0;
            sum += l;
            sum2 += l * l;
        }
        const double mean = sum / n, var = sum2 / n - mean * mean;
        std::printf("%s : %ux%u, %d image(s), luminance moyenne %.3f, écart-type %.3f, %s\n", out.c_str(), w, h, frames,
                    mean, std::sqrt(std::fmax(var, 0.0)), backend->stats(0.0f).c_str());
        const bool ok = mean > 0.03 && mean < 0.9 && var > 1e-3;
        std::printf("%s\n", ok ? "SUCCÈS" : "ÉCHEC : image anormale");
        return ok ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("ERREUR : %s\n", e.what());
        return 1;
    }
}
