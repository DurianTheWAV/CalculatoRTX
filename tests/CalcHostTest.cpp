// CalculatoRTX - tests unitaires du cœur de calcul exécutés sur le CPU.
//
// Le code testé (src/calc/CalcCore.cuh) est exactement celui qu'exécute le kernel GPU CUDA :
// il est simplement instancié ici pour l'hôte, ce qui permet de valider l'analyseur,
// l'arithmétique double-double et le formatage sans carte graphique.
#include "../src/calc/CalcCore.cuh"
#include "CalcCases.h"

#include <cstdio>
#include <cstring>
#include <string>

using namespace crtx;
using namespace crtx::calc;

namespace {

// Convertit le texte de l'afficheur (glyphes spéciaux) en ASCII lisible.
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

Result run(const char* program, AngleMode mode, double ans = 0.0, double mem = 0.0)
{
    Request rq{};
    std::strncpy(rq.program, program, kMaxProgram - 1);
    rq.length = static_cast<int>(std::strlen(rq.program));
    rq.angleMode = mode;
    rq.ansHi = ans;
    rq.memHi = mem;
    Result r{};
    core::evaluateSequential(rq, r);
    return r;
}

int g_failures = 0;

void expectText(const char* program, AngleMode mode, const char* expected)
{
    const Result r = run(program, mode);
    const std::string got = r.status == kOk ? toAscii(r.text) : ("<status " + std::to_string(r.status) + ">");
    const bool ok = got == expected;
    if (!ok) ++g_failures;
    std::printf("%s  %-18s -> %-26s (attendu %s)  dd=%.17g%+.3g  f64=%.17g  iv=[%.17g, %.17g] ok=%d\n",
                ok ? "OK  " : "FAIL", program, got.c_str(), expected, r.ddHi, r.ddLo, r.f64, r.ivLo, r.ivHi,
                r.consistent);
}

void expectStatus(const char* program, AngleMode mode, int status)
{
    const Result r = run(program, mode);
    const bool ok = r.status == status;
    if (!ok) ++g_failures;
    std::printf("%s  %-18s -> statut %d (attendu %d)\n", ok ? "OK  " : "FAIL", program, r.status, status);
}

}  // namespace

int main()
{
    for (const tests::TextCase& c : tests::kTextCases) expectText(c.program, c.mode, c.expected);
    for (const tests::StatusCase& c : tests::kStatusCases) expectStatus(c.program, c.mode, c.status);

    std::printf("\n%s : %d échec(s)\n", g_failures == 0 ? "SUCCÈS" : "ÉCHEC", g_failures);
    return g_failures == 0 ? 0 : 1;
}
