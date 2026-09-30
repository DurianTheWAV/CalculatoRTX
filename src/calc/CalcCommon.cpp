// CalculatoRTX - parties communes aux moteurs de calcul GPU (CUDA et Vulkan).
#include "ICalcEngine.h"

#include "../common/Check.h"

#include <cmath>
#include <cstring>

namespace crtx {
namespace calc {

Request makeRequest(const std::string& program, AngleMode mode, double ansHi, double ansLo, double memHi,
                    double memLo)
{
    Request rq;
    std::memset(&rq, 0, sizeof(rq));
    const size_t n = program.size() < static_cast<size_t>(kMaxProgram) ? program.size() : kMaxProgram;
    std::memcpy(rq.program, program.data(), n);
    rq.length = static_cast<int>(n);
    rq.angleMode = mode;
    rq.ansHi = ansHi;
    rq.ansLo = ansLo;
    rq.memHi = memHi;
    rq.memLo = memLo;
    return rq;
}

bool runSelfTest(ICalcEngine& engine)
{
    struct Case {
        const char* program;
        AngleMode mode;
        double expected;
    };
    const Case cases[] = {
        {"1+2*3", kDegrees, 7.0},
        {"2^10", kDegrees, 1024.0},
        {"s(30)", kDegrees, 0.5},
        {"c(p)", kRadians, -1.0},
        {"q(2)w", kDegrees, 2.0},
        {"5!", kDegrees, 120.0},
        {"-2^2", kDegrees, -4.0},
        {"2(3+4)", kDegrees, 14.0},
        {"L(1000)", kDegrees, 3.0},
        {"3R27", kDegrees, 3.0},
        {"1.5E3/3", kDegrees, 500.0},
        {"T(1)", kDegrees, 45.0},
    };
    bool ok = true;
    for (const Case& c : cases) {
        const Result r = engine.evaluate(c.program, c.mode, 0.0, 0.0, 0.0, 0.0);
        const double v = r.ddHi + r.ddLo;
        const bool pass = r.status == kOk && std::fabs(v - c.expected) <= 1e-12 * std::fmax(1.0, std::fabs(c.expected));
        if (!pass) {
            CRTX_LOG("Auto-test GPU (%s) ÉCHEC : '%s' -> statut %d, valeur %.17g (attendu %.17g)", engine.apiName(),
                     c.program, r.status, v, c.expected);
            ok = false;
        }
    }
    return ok;
}

}  // namespace calc
}  // namespace crtx
