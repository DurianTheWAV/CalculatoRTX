// CalculatoRTX - backend NVIDIA : CUDA + OptiX + DLSS + débruiteur IA.
//
// Chronologie d'une image avec DLSS (valeurs du sémaphore timeline partagé CUDA/Vulkan, base = 4*n) :
//   CUDA   : path tracing OptiX -> débruiteur IA -> copie des entrées DLSS -> signal(base+1)
//   Vulkan : attend(base+1) -> copies tampon->image -> DLSS -> copie image->tampon -> signal(base+2)
//   CUDA   : attend(base+2) -> post-traitement (CUDA Graph) -> signal(base+3)
//   Vulkan : attend(base+3) -> copie vers la swapchain -> présentation -> signal(base+4)
// Sans DLSS (image immobile, accumulation native), les étapes base+1/base+2 sont sautées.
#pragma once

#include "Backend.h"

#include <memory>

namespace crtx {


// Crée le backend NVIDIA ; lève une exception s'il n'y a pas de GPU CUDA / OptiX utilisable.
std::unique_ptr<Backend> createNvidiaBackend(GLFWwindow* window, const BackendOptions& opt);

}  // namespace crtx
