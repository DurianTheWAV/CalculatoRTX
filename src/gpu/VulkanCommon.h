// CalculatoRTX - inclusion multiplateforme de Vulkan + GLFW.
#pragma once

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#ifndef VK_USE_PLATFORM_WIN32_KHR
#define VK_USE_PLATFORM_WIN32_KHR
#endif
#endif

#include <vulkan/vulkan.h>

#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "../common/Check.h"

#define VK_CHECK(call)                                                                             \
    do {                                                                                           \
        VkResult crtxVk_ = (call);                                                                 \
        if (crtxVk_ != VK_SUCCESS)                                                                 \
            ::crtx::throwError(std::string("Vulkan: erreur ") + std::to_string(crtxVk_) + " [" #call "]", \
                               __FILE__, __LINE__);                                                \
    } while (0)
