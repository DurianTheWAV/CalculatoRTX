// CalculatoRTX - moteur de calcul GPU Vulkan.
#include "VkCalcEngine.h"

#include "VkShaders.h"

#include <cstring>

namespace crtx {

using namespace calc;

VkCalcEngine::VkCalcEngine(VulkanContext& vk) : vk_(vk)
{
    VkDevice dev = vk_.device();
    pipe_ = createComputePipeline(dev, g_spvCalcBlob(),
                                  {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER}, 0);
    const VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    request_ = vk_.createBuffer(sizeof(Request), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);
    result_ = vk_.createBuffer(sizeof(Result), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host);

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &pool_));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &pipe_.setLayout;
    VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &set_));
    VkDescriptorBufferInfo bi[2] = {{request_.buffer, 0, sizeof(Request)}, {result_.buffer, 0, sizeof(Result)}};
    VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    for (int i = 0; i < 2; ++i) {
        w[i].dstSet = set_;
        w[i].dstBinding = static_cast<uint32_t>(i);
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = vk_.computeFamily();
    VK_CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cmdPool_));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool_;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cmd_));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(dev, &fci, nullptr, &fence_));
    if (vk_.timestampBits(vk_.computeFamily()) > 0) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = 2;
        VK_CHECK(vkCreateQueryPool(dev, &qi, nullptr, &queries_));
    }
    CRTX_LOG("Moteur de calcul Vulkan : compute shader FP64 sur file %s",
             vk_.computeQueue() != vk_.queue() ? "de calcul asynchrone" : "principale");
}

VkCalcEngine::~VkCalcEngine()
{
    VkDevice dev = vk_.device();
    vkQueueWaitIdle(vk_.computeQueue());
    if (queries_) vkDestroyQueryPool(dev, queries_, nullptr);
    if (fence_) vkDestroyFence(dev, fence_, nullptr);
    if (cmdPool_) vkDestroyCommandPool(dev, cmdPool_, nullptr);
    if (pool_) vkDestroyDescriptorPool(dev, pool_, nullptr);
    vk_.destroyBuffer(request_);
    vk_.destroyBuffer(result_);
    destroyComputePipeline(dev, pipe_);
}

Result VkCalcEngine::evaluate(const std::string& program, AngleMode mode, double ansHi, double ansLo, double memHi,
                              double memLo, float* gpuMicroseconds)
{
    const Request rq = makeRequest(program, mode, ansHi, ansLo, memHi, memLo);
    std::memcpy(request_.mapped, &rq, sizeof(rq));

    VK_CHECK(vkResetCommandBuffer(cmd_, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd_, &bi));
    if (queries_) {
        vkCmdResetQueryPool(cmd_, queries_, 0, 2);
        vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, 0);
    }
    VulkanContext::memoryBarrier(cmd_, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_WRITE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    vkCmdBindPipeline(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_.pipeline);
    vkCmdBindDescriptorSets(cmd_, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_.layout, 0, 1, &set_, 0, nullptr);
    vkCmdDispatch(cmd_, 1, 1, 1);
    VulkanContext::memoryBarrier(cmd_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
    if (queries_) vkCmdWriteTimestamp(cmd_, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 1);
    VK_CHECK(vkEndCommandBuffer(cmd_));

    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd_;
    VK_CHECK(vkResetFences(vk_.device(), 1, &fence_));
    VK_CHECK(vkQueueSubmit(vk_.computeQueue(), 1, &si, fence_));
    VK_CHECK(vkWaitForFences(vk_.device(), 1, &fence_, VK_TRUE, UINT64_MAX));

    Result r;
    std::memcpy(&r, result_.mapped, sizeof(r));
    if (gpuMicroseconds) {
        *gpuMicroseconds = 0.0f;
        uint64_t ts[2] = {0, 0};
        if (queries_ && vkGetQueryPoolResults(vk_.device(), queries_, 0, 2, sizeof(ts), ts, sizeof(uint64_t),
                                              VK_QUERY_RESULT_64_BIT) == VK_SUCCESS)
            *gpuMicroseconds = static_cast<float>(static_cast<double>(ts[1] - ts[0]) * vk_.timestampPeriodNs() * 1e-3);
    }
    return r;
}

}  // namespace crtx
