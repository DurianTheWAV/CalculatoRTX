// CalculatoRTX - cas de test du moteur de calcul, partagés par les tests CPU (CalcHostTest)
// et GPU Vulkan (CalcVulkanTest).
#pragma once

#include "../src/calc/CalcTypes.h"

namespace crtx {
namespace calc {
namespace tests {

struct TextCase {
    const char* program;
    AngleMode mode;
    const char* expected;  // texte de l'afficheur, glyphes convertis en ASCII ('-', 'x')
};

struct StatusCase {
    const char* program;
    AngleMode mode;
    int status;
};

inline const TextCase kTextCases[] = {
    // Arithmétique et priorités
    {"1+2*3", kDegrees, "7"},
    {"(1+2)*3", kDegrees, "9"},
    {"2^10", kDegrees, "1024"},
    {"2^3^2", kDegrees, "512"},
    {"-2^2", kDegrees, "-4"},
    {"2^-3", kDegrees, "0.125"},
    {"2(3+4)", kDegrees, "14"},
    {"2p", kRadians, "6.28318530717959"},
    {"1/3", kDegrees, "0.33333333333333"},
    {"2/3", kDegrees, "0.66666666666667"},
    {"0.1+0.2", kDegrees, "0.3"},
    {"1E20", kDegrees, "1x10{20}"},
    {"1.5E3/3", kDegrees, "500"},
    {"123456789*987654321", kDegrees, "1.219326311x10{17}"},
    {"1E-7*3", kDegrees, "3x10{-7}"},
    {"0.000001234", kDegrees, "0.000001234"},
    {"~5+3", kDegrees, "-2"},
    // Fonctions
    {"s(30)", kDegrees, "0.5"},
    {"c(60)", kDegrees, "0.5"},
    {"t(45)", kDegrees, "1"},
    {"s(p)", kRadians, "0"},
    {"c(p)", kRadians, "-1"},
    {"s(1)", kRadians, "0.8414709848079"},
    {"S(0.5)", kDegrees, "30"},
    {"T(1)", kDegrees, "45"},
    {"C(0)", kRadians, "1.5707963267949"},
    {"q(2)", kDegrees, "1.4142135623731"},
    {"q(2)w", kDegrees, "2"},
    {"Q(27)", kDegrees, "3"},
    {"Q(~8)", kDegrees, "-2"},
    {"3R27", kDegrees, "3"},
    {"l(e)", kDegrees, "1"},
    {"L(1000)", kDegrees, "3"},
    {"x(1)", kDegrees, "2.71828182845905"},
    {"X(3)", kDegrees, "1000"},
    {"5!", kDegrees, "120"},
    {"20!", kDegrees, "2.432902008x10{18}"},
    {"15!", kDegrees, "1307674368000"},
    {"170!", kDegrees, "7.257415615x10{306}"},
    {"4i", kDegrees, "0.25"},
    {"3W", kDegrees, "27"},
    {"s(30)w+c(30)w", kDegrees, "1"},
    {"2^0.5", kDegrees, "1.4142135623731"},
    // Parenthèses non fermées (fermeture automatique)
    {"s(30", kDegrees, "0.5"},
    {"2*(3+4", kDegrees, "14"},
};

inline const StatusCase kStatusCases[] = {
    {"1/0", kDegrees, kDivByZero},
    {"q(~1)", kDegrees, kDomainError},
    {"l(0)", kDegrees, kDomainError},
    {"t(90)", kDegrees, kDomainError},
    {"S(2)", kDegrees, kDomainError},
    {"1+", kDegrees, kSyntaxError},
    {"*2", kDegrees, kSyntaxError},
    {"()", kDegrees, kSyntaxError},
    {"", kDegrees, kEmpty},
    {"10^400", kDegrees, kOverflow},
    {"171!", kDegrees, kOverflow},
};

}  // namespace tests
}  // namespace calc
}  // namespace crtx
