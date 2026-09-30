// CalculatoRTX - backend Vulkan : ray tracing Vulkan (VK_KHR_ray_query), débruiteur à-trous,
// AMD FidelityFX Super Resolution 1 et calcul FP64 en compute shader.
// GPU pris en charge : AMD Radeon RX 6000 / 7000 / 9000 (RDNA 2+), Intel Arc (sans calcul FP64
// natif : voir README), NVIDIA RTX, et tout pilote Vulkan 1.2 offrant le ray query.
#pragma once

#include "Backend.h"

#include <memory>

namespace crtx {

// window peut être nul (rendu hors écran pour les tests) : largeur/hauteur fixent alors la taille.
std::unique_ptr<Backend> createVulkanBackend(GLFWwindow* window, const BackendOptions& opt);

}  // namespace crtx
