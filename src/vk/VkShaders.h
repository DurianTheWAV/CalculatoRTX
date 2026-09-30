// CalculatoRTX - SPIR-V des shaders du backend Vulkan, compilés par glslangValidator à la
// construction et intégrés à l'exécutable (fichiers générés par cmake/EmbedFile.cmake).
#pragma once

#include "VkPipeline.h"

#include <cstddef>

#define CRTX_DECLARE_SPIRV(name)                  \
    extern "C" const unsigned char name[];        \
    extern "C" const std::size_t name##Size;      \
    inline ::crtx::SpirvBlob name##Blob() { return ::crtx::SpirvBlob{name, name##Size}; }

CRTX_DECLARE_SPIRV(g_spvCalc)
CRTX_DECLARE_SPIRV(g_spvEnvBake)
CRTX_DECLARE_SPIRV(g_spvPathTrace)
CRTX_DECLARE_SPIRV(g_spvPick)
CRTX_DECLARE_SPIRV(g_spvDenoise)
CRTX_DECLARE_SPIRV(g_spvBloomDown)
CRTX_DECLARE_SPIRV(g_spvBloomBlur)
CRTX_DECLARE_SPIRV(g_spvComposite)
CRTX_DECLARE_SPIRV(g_spvFsrEasu)
CRTX_DECLARE_SPIRV(g_spvFsrRcas)
CRTX_DECLARE_SPIRV(g_spvPack)
CRTX_DECLARE_SPIRV(g_spvTemporal)

#undef CRTX_DECLARE_SPIRV
