// CalculatoRTX - moteur de calcul exécuté intégralement sur le GPU (CUDA).
#pragma once

#include "CalcTypes.h"

#include <cuda_runtime.h>

#include <string>

namespace crtx {
namespace calc {

class CalcEngine {
public:
    CalcEngine();
    ~CalcEngine();
    CalcEngine(const CalcEngine&) = delete;
    CalcEngine& operator=(const CalcEngine&) = delete;

    // Envoie le programme au GPU, attend le résultat (quelques dizaines de µs).
    Result evaluate(const std::string& program, AngleMode mode, double ansHi, double ansLo,
                    double memHi, double memLo, float* gpuMicroseconds = nullptr);

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t evStart_ = nullptr;
    cudaEvent_t evStop_ = nullptr;
    Request* hostReq_ = nullptr;  // mémoire hôte verrouillée (pinned)
    Result* hostRes_ = nullptr;
    Request* devReq_ = nullptr;
    Result* devRes_ = nullptr;
};

// Auto-test exécuté au démarrage (vérifie quelques expressions de référence sur GPU).
bool runSelfTest(CalcEngine& engine);

}  // namespace calc
}  // namespace crtx
