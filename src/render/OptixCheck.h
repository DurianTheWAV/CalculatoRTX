// CalculatoRTX - vérification des appels OptiX.
#pragma once

#include "../common/Check.h"

#include <optix.h>

#define OPTIX_CHECK(call)                                                                          \
    do {                                                                                           \
        OptixResult crtxRes_ = (call);                                                             \
        if (crtxRes_ != OPTIX_SUCCESS)                                                             \
            ::crtx::throwError(std::string("OptiX: ") + optixGetErrorName(crtxRes_) + " - " +      \
                                   optixGetErrorString(crtxRes_) + " [" #call "]",                 \
                               __FILE__, __LINE__);                                                \
    } while (0)

#define OPTIX_CHECK_LOG(call)                                                                      \
    do {                                                                                           \
        char log[8192];                                                                            \
        size_t logSize = sizeof(log);                                                              \
        OptixResult crtxRes_ = (call);                                                             \
        if (logSize > 1 && crtxRes_ != OPTIX_SUCCESS) std::fprintf(stderr, "%s\n", log);           \
        if (crtxRes_ != OPTIX_SUCCESS)                                                             \
            ::crtx::throwError(std::string("OptiX: ") + optixGetErrorName(crtxRes_) + " [" #call "]", \
                               __FILE__, __LINE__);                                                \
    } while (0)
