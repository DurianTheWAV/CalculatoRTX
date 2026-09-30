// CalculatoRTX - test du moteur de calcul GPU Vulkan (compute shader FP64).
//
// Chaque expression est évaluée par le GPU (VkCalcEngine) et par le cœur CPU de référence
// (src/calc/CalcCore.cuh, identique au kernel CUDA) : statuts, textes de l'afficheur et
// valeurs double-double doivent concorder. Plusieurs milliers d'expressions aléatoires
// complètent les cas fixes. Sans GPU Vulkan compatible FP64, le test est ignoré (code 77).
// Fonctionne aussi avec le pilote logiciel Mesa lavapipe (VK_ICD_FILENAMES=.../lvp_icd.json).
#include "../src/calc/CalcCore.cuh"
#include "../src/vk/VkCalcEngine.h"
#include "CalcCases.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>

using namespace crtx;
using namespace crtx::calc;

namespace {

std::string toAscii(const char* s)
{
    std::string out;
    for (const unsigned char* p = reinterpret_cast<const unsigned char*>(s); *p; ++p) {
        if (*p == glyph::kMinus) out += '-';
        else if (*p == glyph::kTimes) out += 'x';
        else out += static_cast<char>(*p);
    }
    return out;
}

Result cpuEval(const std::string& program, AngleMode mode, double ans)
{
    Request rq = makeRequest(program, mode, ans, 0.0, 0.0, 0.0);
    Result r{};
    core::evaluateSequential(rq, r);
    return r;
}

int g_failures = 0;
int g_checked = 0;

bool compare(ICalcEngine& gpu, const std::string& program, AngleMode mode, double ans, bool verbose)
{
    const Result g = gpu.evaluate(program, mode, ans, 0.0, 0.0, 0.0);
    const Result c = cpuEval(program, mode, ans);
    ++g_checked;
    bool ok = g.status == c.status;
    std::string why;
    if (!ok) why = "statut GPU " + std::to_string(g.status) + " / CPU " + std::to_string(c.status);
    if (ok && c.status == kOk) {
        const std::string tg = toAscii(g.text), tc = toAscii(c.text);
        const double vg = g.ddHi + g.ddLo, vc = c.ddHi + c.ddLo;
        const double rel = std::fabs(vg - vc) / std::fmax(1e-300, std::fabs(vc));
        if (tg != tc) {
            ok = false;
            why = "texte GPU '" + tg + "' / CPU '" + tc + "'";
        } else if (rel > 1e-26 && std::fabs(vg - vc) > 1e-300) {
            ok = false;
            char b[128];
            std::snprintf(b, sizeof(b), "écart DD relatif %.3g", rel);
            why = b;
        } else if (g.consistent != c.consistent) {
            ok = false;
            why = "indicateur de cohérence GPU " + std::to_string(g.consistent) + " / CPU " + std::to_string(c.consistent);
        }
    }
    if (!ok) {
        ++g_failures;
        std::printf("FAIL  %-28s (%s)  %s\n", program.c_str(), mode == kDegrees ? "DEG" : "RAD", why.c_str());
    } else if (verbose) {
        std::printf("OK    %-28s -> %s\n", program.c_str(), g.status == kOk ? toAscii(g.text).c_str() : "(erreur)");
    }
    return ok;
}

// Générateur d'expressions aléatoires (déterministe) couvrant l'analyseur et les fonctions
struct Gen {
    unsigned s = 12345u;
    unsigned next()
    {
        s = s * 1664525u + 1013904223u;
        return s >> 8;
    }
    std::string number()
    {
        static const char* nums[] = {"0", "1", "2", "3", "7", "10", "0.5", "1.5", "2.25", "12.75", "100", "0.001",
                                     "3E2", "1E-3", "45", "30", "60", "90", "180", "123.456", "0.1", "9", "4", "16"};
        return nums[next() % (sizeof(nums) / sizeof(nums[0]))];
    }
    std::string expr(int depth)
    {
        const unsigned k = next() % (depth > 3 ? 3u : 10u);
        switch (k) {
            case 0: case 1: case 2: return number();
            case 3: {
                static const char ops[] = {'+', '-', '*', '/', '^'};
                return expr(depth + 1) + ops[next() % 5] + expr(depth + 1);
            }
            case 4: {
                static const char fns[] = {'s', 'c', 't', 'S', 'C', 'T', 'l', 'L', 'q', 'Q', 'x', 'X'};
                return std::string(1, fns[next() % 12]) + "(" + expr(depth + 1) + ")";
            }
            case 5: {
                static const char post[] = {'!', 'w', 'W', 'i'};
                return "(" + expr(depth + 1) + ")" + post[next() % 4];
            }
            case 6: return "~" + expr(depth + 1);
            case 7: return "(" + expr(depth + 1) + ")";
            case 8: return std::string(next() % 2 ? "p" : "e") + expr(depth + 1);
            default: return number() + "R" + expr(depth + 1);
        }
    }
};

}  // namespace

int main(int argc, char** argv)
{
    const bool single = argc > 1 && std::strncmp(argv[1], "--expr=", 7) == 0;
    const int randomCount = argc > 1 && !single ? std::atoi(argv[1]) : 3000;
    try {
        VulkanContext vk;
        vk.createInstance({}, false, true);
        DeviceRequest req;
        req.float64 = true;
        try {
            vk.pickPhysicalDevice(req, -1);
        } catch (const std::exception& e) {
            std::printf("IGNORÉ : aucun GPU Vulkan FP64 (%s)\n", e.what());
            return 77;
        }
        vk.createDevice(nullptr, req);
        VkCalcEngine gpu(vk);
        if (single) {  // diagnostic : détail des 4 voies pour une expression
            for (int m = 0; m < 2; ++m) {
                const AngleMode mode = m ? kDegrees : kRadians;
                const Result g = gpu.evaluate(argv[1] + 7, mode, 0, 0, 0, 0);
                const Result c = cpuEval(argv[1] + 7, mode, 0.0);
                for (const Result* r : {&g, &c})
                    std::printf("%s %s st=%d [%d %d %d %d] dd=%.17g%+.6g f64=%.17g f32=%.9g iv=[%.17g, %.17g] ok=%d '%s'\n",
                                r == &g ? "GPU" : "CPU", m ? "DEG" : "RAD", r->status, r->laneStatus[0],
                                r->laneStatus[1], r->laneStatus[2], r->laneStatus[3], r->ddHi, r->ddLo, r->f64,
                                r->f32, r->ivLo, r->ivHi, r->consistent, toAscii(r->text).c_str());
            }
            return 0;
        }

        for (const tests::TextCase& c : tests::kTextCases) compare(gpu, c.program, c.mode, 0.0, true);
        for (const tests::StatusCase& c : tests::kStatusCases) compare(gpu, c.program, c.mode, 0.0, true);
        compare(gpu, "A*2", kDegrees, 21.0, true);
        if (!runSelfTest(gpu)) ++g_failures;

        Gen gen;
        for (int i = 0; i < randomCount; ++i) {
            const std::string e = gen.expr(0);
            compare(gpu, e, (i & 1) ? kDegrees : kRadians, 0.0, false);
        }
        float us = 0.0f;
        gpu.evaluate("s(30)+2^10*3", kDegrees, 0, 0, 0, 0, &us);
        std::printf("\n%d expressions comparées GPU/CPU, %d échec(s) ; dernière évaluation GPU : %.1f µs\n", g_checked,
                    g_failures, us);
        return g_failures == 0 ? 0 : 1;
    } catch (const std::exception& e) {
        std::printf("ERREUR : %s\n", e.what());
        return 1;
    }
}
