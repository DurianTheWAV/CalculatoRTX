// CalculatoRTX - pipelines compute Vulkan (création à partir du SPIR-V intégré).
#pragma once

#include "../gpu/VulkanContext.h"

#include <cstddef>
#include <vector>

namespace crtx {

// SPIR-V compilé à la construction du projet (glslangValidator) et intégré à l'exécutable.
struct SpirvBlob {
    const unsigned char* data;
    std::size_t size;
};

struct ComputePipeline {
    VkPipeline pipeline = VK_NULL_HANDLE;
    VkPipelineLayout layout = VK_NULL_HANDLE;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;  // peut être nul
};

// bindings : types de descripteurs du set 0 (binding i = bindings[i]) ; pushBytes : constantes de poussée.
ComputePipeline createComputePipeline(VkDevice device, const SpirvBlob& spirv,
                                      const std::vector<VkDescriptorType>& bindings, uint32_t pushBytes);
void destroyComputePipeline(VkDevice device, ComputePipeline& p);

inline uint32_t groupsFor(uint32_t n, uint32_t local) { return (n + local - 1) / local; }

}  // namespace crtx
