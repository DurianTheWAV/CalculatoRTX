// CalculatoRTX - pipelines compute Vulkan.
#include "VkPipeline.h"

#include <cstring>

namespace crtx {

ComputePipeline createComputePipeline(VkDevice device, const SpirvBlob& spirv,
                                      const std::vector<VkDescriptorType>& bindings, uint32_t pushBytes)
{
    ComputePipeline p;
    if (spirv.size == 0 || spirv.size % 4 != 0) throwError("SPIR-V invalide", __FILE__, __LINE__);
    std::vector<uint32_t> code(spirv.size / 4);
    std::memcpy(code.data(), spirv.data, spirv.size);  // alignement garanti sur 4 octets
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = spirv.size;
    smi.pCode = code.data();
    VkShaderModule module = VK_NULL_HANDLE;
    VK_CHECK(vkCreateShaderModule(device, &smi, nullptr, &module));

    if (!bindings.empty()) {
        std::vector<VkDescriptorSetLayoutBinding> b(bindings.size());
        for (size_t i = 0; i < bindings.size(); ++i) {
            b[i].binding = static_cast<uint32_t>(i);
            b[i].descriptorType = bindings[i];
            b[i].descriptorCount = 1;
            b[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
        }
        VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        dli.bindingCount = static_cast<uint32_t>(b.size());
        dli.pBindings = b.data();
        VK_CHECK(vkCreateDescriptorSetLayout(device, &dli, nullptr, &p.setLayout));
    }
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = p.setLayout ? 1u : 0u;
    pli.pSetLayouts = p.setLayout ? &p.setLayout : nullptr;
    pli.pushConstantRangeCount = pushBytes ? 1u : 0u;
    pli.pPushConstantRanges = pushBytes ? &pr : nullptr;
    VK_CHECK(vkCreatePipelineLayout(device, &pli, nullptr, &p.layout));

    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    ci.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    ci.stage.module = module;
    ci.stage.pName = "main";
    ci.layout = p.layout;
    VK_CHECK(vkCreateComputePipelines(device, VK_NULL_HANDLE, 1, &ci, nullptr, &p.pipeline));
    vkDestroyShaderModule(device, module, nullptr);
    return p;
}

void destroyComputePipeline(VkDevice device, ComputePipeline& p)
{
    if (p.pipeline) vkDestroyPipeline(device, p.pipeline, nullptr);
    if (p.layout) vkDestroyPipelineLayout(device, p.layout, nullptr);
    if (p.setLayout) vkDestroyDescriptorSetLayout(device, p.setLayout, nullptr);
    p = ComputePipeline{};
}

}  // namespace crtx
