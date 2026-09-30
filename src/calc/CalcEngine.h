// CalculatoRTX - moteur de calcul exécuté intégralement sur le GPU (CUDA, backend NVIDIA).
#pragma once

#include "ICalcEngine.h"

#include <cuda_runtime.h>

#include <string>

namespace crtx {
namespace calc {

class CalcEngine : public ICalcEngine {
public:
    CalcEngine();
    ~CalcEngine() override;
    CalcEngine(const CalcEngine&) = delete;
    CalcEngine& operator=(const CalcEngine&) = delete;

    Result evaluate(const std::string& program, AngleMode mode, double ansHi, double ansLo, double memHi,
                    double memLo, float* gpuMicroseconds = nullptr) override;
    const char* apiName() const override { return "CUDA"; }

private:
    cudaStream_t stream_ = nullptr;
    cudaEvent_t evStart_ = nullptr;
    cudaEvent_t evStop_ = nullptr;
    Request* hostReq_ = nullptr;  // mémoire hôte verrouillée (pinned)
    Result* hostRes_ = nullptr;
    Request* devReq_ = nullptr;
    Result* devRes_ = nullptr;
};

}  // namespace calc
}  // namespace crtx
