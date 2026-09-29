// CalculatoRTX - tests unitaires du cœur de calcul exécutés sur le CPU.
//
// Le code testé (src/calc/CalcCore.cuh) est exactement celui qu'exécute le kernel GPU :
// il est simplement instancié ici pour l'hôte, ce qui permet de valider l'analyseur,
// l'arithmétique double-double et le formatage sans carte graphique.
#include "../src/calc/CalcCore.cuh"

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
    // Arithmétique et priorités
    expectText("1+2*3", kDegrees, "7");
    expectText("(1+2)*3", kDegrees, "9");
    expectText("2^10", kDegrees, "1024");
    expectText("2^3^2", kDegrees, "512");
    expectText("-2^2", kDegrees, "-4");
    expectText("2^-3", kDegrees, "0.125");
    expectText("2(3+4)", kDegrees, "14");
    expectText("2p", kRadians, "6.28318530717959");
    expectText("1/3", kDegrees, "0.33333333333333");
    expectText("2/3", kDegrees, "0.66666666666667");
    expectText("0.1+0.2", kDegrees, "0.3");
    expectText("1E20", kDegrees, "1x10{20}");
    expectText("1.5E3/3", kDegrees, "500");
    expectText("123456789*987654321", kDegrees, "1.219326311x10{17}");
    expectText("1E-7*3", kDegrees, "3x10{-7}");
    expectText("0.000001234", kDegrees, "0.000001234");
    expectText("~5+3", kDegrees, "-2");
    // Fonctions
    expectText("s(30)", kDegrees, "0.5");
    expectText("c(60)", kDegrees, "0.5");
    expectText("t(45)", kDegrees, "1");
    expectText("s(p)", kRadians, "0");
    expectText("c(p)", kRadians, "-1");
    expectText("s(1)", kRadians, "0.8414709848079");
    expectText("S(0.5)", kDegrees, "30");
    expectText("T(1)", kDegrees, "45");
    expectText("C(0)", kRadians, "1.5707963267949");
    expectText("q(2)", kDegrees, "1.4142135623731");
    expectText("q(2)w", kDegrees, "2");
    expectText("Q(27)", kDegrees, "3");
    expectText("Q(~8)", kDegrees, "-2");
    expectText("3R27", kDegrees, "3");
    expectText("l(e)", kDegrees, "1");
    expectText("L(1000)", kDegrees, "3");
    expectText("x(1)", kDegrees, "2.71828182845905");
    expectText("X(3)", kDegrees, "1000");
    expectText("5!", kDegrees, "120");
    expectText("20!", kDegrees, "2.432902008x10{18}");
    expectText("15!", kDegrees, "1307674368000");
    expectText("170!", kDegrees, "7.257415615x10{306}");
    expectText("4i", kDegrees, "0.25");
    expectText("3W", kDegrees, "27");
    expectText("s(30)w+c(30)w", kDegrees, "1");
    expectText("2^0.5", kDegrees, "1.4142135623731");
    // Erreurs
    expectStatus("1/0", kDegrees, kDivByZero);
    expectStatus("q(~1)", kDegrees, kDomainError);
    expectStatus("l(0)", kDegrees, kDomainError);
    expectStatus("t(90)", kDegrees, kDomainError);
    expectStatus("S(2)", kDegrees, kDomainError);
    expectStatus("1+", kDegrees, kSyntaxError);
    expectStatus("*2", kDegrees, kSyntaxError);
    expectStatus("()", kDegrees, kSyntaxError);
    expectStatus("", kDegrees, kEmpty);
    expectStatus("10^400", kDegrees, kOverflow);
    expectStatus("171!", kDegrees, kOverflow);
    // Parenthèses non fermées (fermeture automatique)
    expectText("s(30", kDegrees, "0.5");
    expectText("2*(3+4", kDegrees, "14");

    std::printf("\n%s : %d échec(s)\n", g_failures == 0 ? "SUCCÈS" : "ÉCHEC", g_failures);
    return g_failures == 0 ? 0 : 1;
}
