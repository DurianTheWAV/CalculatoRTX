// CalculatoRTX - macros de vérification d'erreurs (CUDA / OptiX / Vulkan / NGX)
#pragma once

#if defined(CRTX_WITH_CUDA) || defined(__CUDACC__)
#include <cuda_runtime.h>
#endif

#include <cstdio>
#include <sstream>
#include <stdexcept>
#include <string>

namespace crtx {

[[noreturn]] inline void throwError(const std::string& what, const char* file, int line)
{
    std::ostringstream s;
    s << what << "  (" << file << ":" << line << ")";
    throw std::runtime_error(s.str());
}

}  // namespace crtx

#if defined(CRTX_WITH_CUDA) || defined(__CUDACC__)
#define CUDA_CHECK(call)                                                                           \
    do {                                                                                           \
        cudaError_t crtxErr_ = (call);                                                             \
        if (crtxErr_ != cudaSuccess)                                                               \
            ::crtx::throwError(std::string("CUDA: ") + cudaGetErrorName(crtxErr_) + " - " +        \
                                   cudaGetErrorString(crtxErr_) + " [" #call "]",                  \
                               __FILE__, __LINE__);                                                \
    } while (0)

// Vérifie l'erreur de lancement d'un kernel (asynchrone).
#define CUDA_CHECK_LAST() CUDA_CHECK(cudaGetLastError())
#endif

#define CRTX_LOG(...)                                                                              \
    do {                                                                                           \
        std::printf("[CalculatoRTX] " __VA_ARGS__);                                                \
        std::printf("\n");                                                                         \
        std::fflush(stdout);                                                                       \
    } while (0)
