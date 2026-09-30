// CalculatoRTX - moteur de rendu Vulkan (ray query + débruiteur + AMD FSR 1).
#include "VkRenderer.h"

#include "VkShaders.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace crtx {

namespace {

// Miroir de GeometryData (Common.glsl)
struct GpuGeometry {
    uint64_t positions, normals, uvs, indices;
    uint32_t alphaCutout, pad0;
};
static_assert(sizeof(GpuGeometry) == 40, "GpuGeometry : disposition partagée avec les shaders");

constexpr VkMemoryPropertyFlags kHost = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
constexpr VkMemoryPropertyFlags kDevice = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
constexpr VkBufferUsageFlags kAddr = VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
constexpr uint32_t kQueryCount = 3;

VkDeviceSize alignUp(VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) / a * a; }

}  // namespace

VkRenderer::VkRenderer(VulkanContext& vk) : vk_(vk)
{
    VkDevice dev = vk_.device();
    const std::vector<VkDescriptorType> traceBindings = {VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR,
                                                         VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                                         VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER};
    trace_ = createComputePipeline(dev, g_spvPathTraceBlob(), traceBindings, 0);
    pick_ = createComputePipeline(dev, g_spvPickBlob(), traceBindings, 0);
    envBake_ = createComputePipeline(dev, g_spvEnvBakeBlob(), {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}, 0);
    const uint32_t push = sizeof(PostPush);
    denoise_ = createComputePipeline(dev, g_spvDenoiseBlob(), {}, push);
    bloomDown_ = createComputePipeline(dev, g_spvBloomDownBlob(), {}, push);
    bloomBlur_ = createComputePipeline(dev, g_spvBloomBlurBlob(), {}, push);
    composite_ = createComputePipeline(dev, g_spvCompositeBlob(), {}, push);
    easu_ = createComputePipeline(dev, g_spvFsrEasuBlob(), {}, push);
    rcas_ = createComputePipeline(dev, g_spvFsrRcasBlob(), {}, push);
    pack_ = createComputePipeline(dev, g_spvPackBlob(), {}, push);
    temporal_ = createComputePipeline(dev, g_spvTemporalBlob(), {}, sizeof(TemporalPush));

    frameUbo_ = vk_.createBuffer(sizeof(VkFrameParams), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, kHost);
    pickBuf_ = vk_.createBuffer(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | kAddr, kHost);
    *static_cast<int*>(pickBuf_.mapped) = -1;

    createEnvironment();
    createDescriptors();

    if (vk_.timestampBits(vk_.queueFamily()) > 0) {
        VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
        qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
        qi.queryCount = kQueryCount;
        VK_CHECK(vkCreateQueryPool(dev, &qi, nullptr, &queries_));
    }
}

VkRenderer::~VkRenderer()
{
    VkDevice dev = vk_.device();
    vkDeviceWaitIdle(dev);
    beginFrame();
    for (MeshGpu& m : meshes_) freeMesh(m);
    beginFrame();
    if (tlas_) vk_.rt().destroyAS(dev, tlas_, nullptr);
    for (Buffer* b : {&tlasBuffer_, &tlasScratch_, &instanceBuffer_, &materials_, &geometries_, &instances_, &lights_,
                      &frameUbo_, &pickBuf_})
        vk_.destroyBuffer(*b);
    freeFrameBuffers();
    if (queries_) vkDestroyQueryPool(dev, queries_, nullptr);
    if (envSampler_) vkDestroySampler(dev, envSampler_, nullptr);
    vk_.destroyImage(env_);
    if (pool_) vkDestroyDescriptorPool(dev, pool_, nullptr);
    for (ComputePipeline* p : {&trace_, &pick_, &envBake_, &denoise_, &bloomDown_, &bloomBlur_, &composite_, &easu_,
                               &rcas_, &pack_, &temporal_})
        destroyComputePipeline(dev, *p);
}

// ---------------------------------------------------------------- environnement HDR
void VkRenderer::createEnvironment()
{
    VkDevice dev = vk_.device();
    const uint32_t w = 1024, h = 512;
    env_ = vk_.createImage(w, h, VK_FORMAT_R32G32B32A32_SFLOAT, VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    VkSamplerCreateInfo si{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    si.magFilter = VK_FILTER_LINEAR;  // filtrage bilinéaire par les unités de texture
    si.minFilter = VK_FILTER_LINEAR;
    si.addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    si.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    si.maxLod = 0.0f;
    VK_CHECK(vkCreateSampler(dev, &si, nullptr, &envSampler_));

    VkDescriptorPoolSize ps{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = 1;
    pci.pPoolSizes = &ps;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &pool));
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &envBake_.setLayout;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateDescriptorSets(dev, &ai, &set));
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, env_.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet wr{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    wr.dstSet = set;
    wr.dstBinding = 0;
    wr.descriptorCount = 1;
    wr.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    wr.pImageInfo = &ii;
    vkUpdateDescriptorSets(dev, 1, &wr, 0, nullptr);
    vk_.immediateSubmit([&](VkCommandBuffer cmd) {
        VulkanContext::imageBarrier(cmd, env_.image, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL,
                                    VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                    VK_ACCESS_SHADER_WRITE_BIT);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, envBake_.pipeline);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, envBake_.layout, 0, 1, &set, 0, nullptr);
        vkCmdDispatch(cmd, groupsFor(w, 16), groupsFor(h, 16), 1);
        VulkanContext::imageBarrier(cmd, env_.image, VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                    VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
    });
    vkDestroyDescriptorPool(dev, pool, nullptr);
}

void VkRenderer::createDescriptors()
{
    VkDevice dev = vk_.device();
    const VkDescriptorPoolSize sizes[] = {{VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR, 2},
                                          {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 2},
                                          {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2}};
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 2;
    pci.poolSizeCount = 3;
    pci.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &pool_));
    const VkDescriptorSetLayout layouts[2] = {trace_.setLayout, pick_.setLayout};
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = pool_;
    ai.descriptorSetCount = 2;
    ai.pSetLayouts = layouts;
    VkDescriptorSet sets[2];
    VK_CHECK(vkAllocateDescriptorSets(dev, &ai, sets));
    traceSet_ = sets[0];
    pickSet_ = sets[1];
    for (VkDescriptorSet s : sets) {
        VkDescriptorImageInfo ii{envSampler_, env_.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        VkDescriptorBufferInfo bi{frameUbo_.buffer, 0, sizeof(VkFrameParams)};
        VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        w[0].dstSet = s;
        w[0].dstBinding = 1;
        w[0].descriptorCount = 1;
        w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        w[0].pImageInfo = &ii;
        w[1].dstSet = s;
        w[1].dstBinding = 2;
        w[1].descriptorCount = 1;
        w[1].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
        w[1].pBufferInfo = &bi;
        vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
    }
}

// ---------------------------------------------------------------- gestion mémoire
void VkRenderer::deferDestroy(Buffer& b)
{
    if (b.buffer) pendingDestroy_.push_back(b);
    b = Buffer{};
}

void VkRenderer::freeMesh(MeshGpu& m)
{
    if (m.blas) pendingDestroyAs_.push_back(m.blas);
    for (Buffer* b : {&m.positions, &m.normals, &m.uvs, &m.indices, &m.asBuffer}) deferDestroy(*b);
    m = MeshGpu{};
}

void VkRenderer::beginFrame()
{
    VkDevice dev = vk_.device();
    for (VkAccelerationStructureKHR as : pendingDestroyAs_) vk_.rt().destroyAS(dev, as, nullptr);
    pendingDestroyAs_.clear();
    for (Buffer& b : pendingDestroy_) vk_.destroyBuffer(b);
    pendingDestroy_.clear();
    // chronos de l'image précédente
    if (queries_ && queriesWritten_) {
        uint64_t ts[kQueryCount] = {};
        if (vkGetQueryPoolResults(dev, queries_, 0, kQueryCount, sizeof(ts), ts, sizeof(uint64_t),
                                  VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            const double ns = vk_.timestampPeriodNs();
            traceMs_ = static_cast<float>((ts[1] - ts[0]) * ns * 1e-6);
            postMs_ = static_cast<float>((ts[2] - ts[1]) * ns * 1e-6);
        }
    }
}

Buffer VkRenderer::uploadBuffer(VkCommandBuffer cmd, const void* data, VkDeviceSize bytes, VkBufferUsageFlags usage)
{
    Buffer dst = vk_.createBuffer(bytes, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT | kAddr, kDevice);
    Buffer staging = vk_.createBuffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kHost);
    std::memcpy(staging.mapped, data, static_cast<size_t>(bytes));
    const VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, staging.buffer, dst.buffer, 1, &region);
    deferDestroy(staging);  // libéré quand l'image aura fini de s'exécuter
    return dst;
}

void VkRenderer::ensureHostBuffer(Buffer& b, VkDeviceSize bytes, VkBufferUsageFlags usage)
{
    if (b.buffer && b.size >= bytes) return;
    deferDestroy(b);
    b = vk_.createBuffer(std::max<VkDeviceSize>(bytes * 2, 256), usage | kAddr, kHost);
}

// ---------------------------------------------------------------- BLAS des maillages
void VkRenderer::buildMeshes(CalculatorScene& scene, VkCommandBuffer cmd)
{
    auto& meshes = scene.meshes();
    if (meshes_.size() < meshes.size()) meshes_.resize(meshes.size());
    const VkBufferUsageFlags geomUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                         VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR;
    std::vector<size_t> dirty;
    for (size_t i = 0; i < meshes.size(); ++i) {
        if (!meshes[i].dirty) continue;
        meshes[i].dirty = false;
        freeMesh(meshes_[i]);
        const Mesh& m = meshes[i].mesh;
        MeshGpu& g = meshes_[i];
        g.alpha = meshes[i].alphaCutout;
        if (m.empty()) continue;
        g.positions = uploadBuffer(cmd, m.positions.data(), sizeof(float3) * m.positions.size(), geomUsage);
        g.indices = uploadBuffer(cmd, m.indices.data(), sizeof(uint3) * m.indices.size(), geomUsage);
        if (!m.flat && !m.normals.empty())
            g.normals = uploadBuffer(cmd, m.normals.data(), sizeof(float3) * m.normals.size(), geomUsage);
        if (!m.uvs.empty()) g.uvs = uploadBuffer(cmd, m.uvs.data(), sizeof(float2) * m.uvs.size(), geomUsage);
        g.triangles = static_cast<uint32_t>(m.indices.size());
        dirty.push_back(i);
        geometryChanged_ = true;
    }
    if (dirty.empty()) return;
    VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_ACCESS_SHADER_READ_BIT);

    // Toutes les constructions dans un seul appel, chacune avec sa zone de travail
    const RayTracingFns& rt = vk_.rt();
    std::vector<VkAccelerationStructureGeometryKHR> geoms(dirty.size());
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos(dirty.size());
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges(dirty.size());
    std::vector<VkDeviceSize> scratchOffsets(dirty.size());
    const VkDeviceSize scratchAlign = vk_.asScratchAlignment();
    VkDeviceSize scratchTotal = 0;
    for (size_t k = 0; k < dirty.size(); ++k) {
        const Mesh& m = meshes[dirty[k]].mesh;
        MeshGpu& g = meshes_[dirty[k]];
        VkAccelerationStructureGeometryKHR& geo = geoms[k];
        geo = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        geo.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        // grille perforée : non opaque, chaque intersection candidate est testée par le shader
        geo.flags = g.alpha ? VK_GEOMETRY_NO_DUPLICATE_ANY_HIT_INVOCATION_BIT_KHR : VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto& tri = geo.geometry.triangles;
        tri.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        tri.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        tri.vertexData.deviceAddress = g.positions.address;
        tri.vertexStride = sizeof(float3);
        tri.maxVertex = static_cast<uint32_t>(m.positions.size() - 1);
        tri.indexType = VK_INDEX_TYPE_UINT32;
        tri.indexData.deviceAddress = g.indices.address;
        VkAccelerationStructureBuildGeometryInfoKHR& bi = infos[k];
        bi = VkAccelerationStructureBuildGeometryInfoKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
        bi.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
        bi.geometryCount = 1;
        bi.pGeometries = &geo;
        VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        rt.buildSizes(vk_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &g.triangles, &sz);
        g.asBuffer = vk_.createBuffer(sz.accelerationStructureSize,
                                      VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | kAddr, kDevice);
        VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        ci.buffer = g.asBuffer.buffer;
        ci.size = sz.accelerationStructureSize;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
        VK_CHECK(rt.createAS(vk_.device(), &ci, nullptr, &g.blas));
        VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
        ai.accelerationStructure = g.blas;
        g.blasAddress = rt.asAddress(vk_.device(), &ai);
        bi.dstAccelerationStructure = g.blas;
        scratchOffsets[k] = scratchTotal;
        scratchTotal = alignUp(scratchTotal + sz.buildScratchSize, scratchAlign);
        ranges[k] = VkAccelerationStructureBuildRangeInfoKHR{g.triangles, 0, 0, 0};
    }
    Buffer scratch = vk_.createBuffer(scratchTotal + scratchAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | kAddr, kDevice);
    const VkDeviceAddress base = alignUp(scratch.address, scratchAlign);
    std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> rangePtrs(dirty.size());
    for (size_t k = 0; k < dirty.size(); ++k) {
        infos[k].pGeometries = &geoms[k];
        infos[k].scratchData.deviceAddress = base + scratchOffsets[k];
        rangePtrs[k] = &ranges[k];
    }
    rt.cmdBuild(cmd, static_cast<uint32_t>(infos.size()), infos.data(), rangePtrs.data());
    deferDestroy(scratch);
    VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                                 VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

// ---------------------------------------------------------------- TLAS + données par instance
void VkRenderer::buildTlas(const CalculatorScene& scene, VkCommandBuffer cmd)
{
    const auto& instances = scene.instances();
    if (prevTransforms_.size() != instances.size()) {
        prevTransforms_.resize(instances.size());
        for (size_t i = 0; i < instances.size(); ++i) prevTransforms_[i] = instances[i].transform;
    }
    std::vector<VkAccelerationStructureInstanceKHR> vi;
    std::vector<InstanceData> idata;
    vi.reserve(instances.size());
    idata.reserve(instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        const SceneInstance& si = instances[i];
        if (si.mesh < 0 || si.mesh >= static_cast<int>(meshes_.size())) continue;
        const MeshGpu& g = meshes_[si.mesh];
        if (!g.blas) continue;  // maillage vide (ex. texte vide)
        VkAccelerationStructureInstanceKHR x{};
        std::memcpy(&x.transform, si.transform.m, sizeof(x.transform));
        x.instanceCustomIndex = static_cast<uint32_t>(idata.size());
        x.mask = si.mask & 0xFFu;
        x.instanceShaderBindingTableRecordOffset = 0;
        x.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
        x.accelerationStructureReference = g.blasAddress;
        vi.push_back(x);
        InstanceData d{};
        std::memcpy(d.prevObjectToWorld, prevTransforms_[i].m, sizeof(d.prevObjectToWorld));
        d.geometry = si.mesh;
        d.material = si.material;
        d.pickId = si.pickId;
        d.lightIndex = si.lightIndex;
        d.glow = si.glow;
        d.glowColor = si.glowColor;
        idata.push_back(d);
        prevTransforms_[i] = si.transform;
    }
    const size_t n = std::max<size_t>(vi.size(), 1);
    ensureHostBuffer(instanceBuffer_, sizeof(VkAccelerationStructureInstanceKHR) * n,
                     VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR);
    std::memcpy(instanceBuffer_.mapped, vi.data(), sizeof(VkAccelerationStructureInstanceKHR) * vi.size());
    ensureHostBuffer(instances_, sizeof(InstanceData) * n, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(instances_.mapped, idata.data(), sizeof(InstanceData) * idata.size());

    const RayTracingFns& rt = vk_.rt();
    VkAccelerationStructureGeometryKHR geo{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    geo.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    geo.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    geo.geometry.instances.arrayOfPointers = VK_FALSE;
    geo.geometry.instances.data.deviceAddress = instanceBuffer_.address;
    VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
    bi.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = &geo;
    const uint32_t count = static_cast<uint32_t>(vi.size());
    VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    rt.buildSizes(vk_.device(), VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &count, &sz);
    if (!tlas_ || sz.accelerationStructureSize > tlasCapacity_) {
        if (tlas_) pendingDestroyAs_.push_back(tlas_);
        deferDestroy(tlasBuffer_);
        tlasCapacity_ = sz.accelerationStructureSize * 2;
        tlasBuffer_ = vk_.createBuffer(tlasCapacity_, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | kAddr, kDevice);
        VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
        ci.buffer = tlasBuffer_.buffer;
        ci.size = tlasCapacity_;
        ci.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
        VK_CHECK(rt.createAS(vk_.device(), &ci, nullptr, &tlas_));
        tlasDescriptorDirty_ = true;
    }
    const VkDeviceSize align = vk_.asScratchAlignment();
    if (!tlasScratch_.buffer || tlasScratch_.size < sz.buildScratchSize + align) {
        deferDestroy(tlasScratch_);
        tlasScratch_ = vk_.createBuffer((sz.buildScratchSize + align) * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | kAddr, kDevice);
    }
    bi.dstAccelerationStructure = tlas_;
    bi.scratchData.deviceAddress = alignUp(tlasScratch_.address, align);
    const VkAccelerationStructureBuildRangeInfoKHR range{count, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* rp = &range;
    rt.cmdBuild(cmd, 1, &bi, &rp);
    VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                 VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                 VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);

    if (tlasDescriptorDirty_) {
        for (VkDescriptorSet s : {traceSet_, pickSet_}) {
            VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
            asw.accelerationStructureCount = 1;
            asw.pAccelerationStructures = &tlas_;
            VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            w.pNext = &asw;
            w.dstSet = s;
            w.dstBinding = 0;
            w.descriptorCount = 1;
            w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
            vkUpdateDescriptorSets(vk_.device(), 1, &w, 0, nullptr);
        }
        tlasDescriptorDirty_ = false;
    }
}

void VkRenderer::syncScene(CalculatorScene& scene, VkCommandBuffer cmd)
{
    buildMeshes(scene, cmd);
    // table des géométries (adresses GPU), matériaux, luminaires
    std::vector<GpuGeometry> geo(meshes_.size());
    for (size_t i = 0; i < meshes_.size(); ++i) {
        geo[i] = GpuGeometry{meshes_[i].positions.address, meshes_[i].normals.address, meshes_[i].uvs.address,
                             meshes_[i].indices.address, meshes_[i].alpha ? 1u : 0u, 0u};
    }
    ensureHostBuffer(geometries_, sizeof(GpuGeometry) * std::max<size_t>(geo.size(), 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(geometries_.mapped, geo.data(), sizeof(GpuGeometry) * geo.size());
    const auto& mats = scene.materials();
    ensureHostBuffer(materials_, sizeof(Material) * std::max<size_t>(mats.size(), 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    std::memcpy(materials_.mapped, mats.data(), sizeof(Material) * mats.size());
    const auto& lights = scene.lights();
    ensureHostBuffer(lights_, sizeof(RectLight) * std::max<size_t>(lights.size(), 1), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    if (!lights.empty()) std::memcpy(lights_.mapped, lights.data(), sizeof(RectLight) * lights.size());
    numLights_ = static_cast<uint32_t>(lights.size());
    buildTlas(scene, cmd);
    if (geometryChanged_) accumCount_ = 0;
    geometryChanged_ = false;
}

// ---------------------------------------------------------------- tampons d'image
void VkRenderer::freeFrameBuffers()
{
    for (Buffer* b : {&color_, &albedo_, &normal_, &denoiseA_, &denoiseB_, &bloomA_, &bloomB_, &ldr_, &easuOut_,
                      &rcasOut_, &present_, &motion_, &hist_[0], &hist_[1], &histDepth_[0], &histDepth_[1]})
        if (b->buffer) vk_.destroyBuffer(*b);
    historyValid_ = false;
}

void VkRenderer::resize(uint32_t displayW, uint32_t displayH)
{
    displayW = std::max(1u, displayW);
    displayH = std::max(1u, displayH);
    if (displayW == displayW_ && displayH == displayH_ && color_.buffer) return;
    vkDeviceWaitIdle(vk_.device());
    freeFrameBuffers();
    displayW_ = displayW;
    displayH_ = displayH;
    renderW_ = displayW;
    renderH_ = displayH;
    // Tampons de rendu dimensionnés pour la résolution native : les images en mouvement
    // (FSR 1) n'en utilisent qu'une partie, sans réallocation.
    const VkBufferUsageFlags u = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | kAddr;
    const VkDeviceSize dn = static_cast<VkDeviceSize>(displayW) * displayH;
    for (Buffer* b : {&color_, &albedo_, &normal_, &denoiseA_, &denoiseB_, &ldr_, &easuOut_, &rcasOut_, &hist_[0], &hist_[1]})
        *b = vk_.createBuffer(dn * 16, u, kDevice);
    motion_ = vk_.createBuffer(dn * 8, u, kDevice);
    for (Buffer* b : {&histDepth_[0], &histDepth_[1]}) *b = vk_.createBuffer(dn * 4, u, kDevice);
    const VkDeviceSize bn = static_cast<VkDeviceSize>((displayW + 3) / 4) * ((displayH + 3) / 4);
    for (Buffer* b : {&bloomA_, &bloomB_}) *b = vk_.createBuffer(bn * 16, u, kDevice);
    present_ = vk_.createBuffer(dn * 4, u | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, kDevice);
    accumCount_ = 0;
}

void VkRenderer::post(VkCommandBuffer cmd, const ComputePipeline& p, const PostPush& push, uint32_t w, uint32_t h,
                      uint32_t local)
{
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, p.pipeline);
    vkCmdPushConstants(cmd, p.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PostPush), &push);
    vkCmdDispatch(cmd, groupsFor(w, local), groupsFor(h, local), 1);
    VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

int VkRenderer::pickResult() const { return *static_cast<const int*>(pickBuf_.mapped); }

// ---------------------------------------------------------------- image complète
void VkRenderer::writeParams(const VkFrameDesc& f)
{
    VkFrameParams p{};
    p.color = color_.address;
    p.albedo = albedo_.address;
    p.normal = normal_.address;
    p.motion = motion_.address;
    p.materials = materials_.address;
    p.geometries = geometries_.address;
    p.instances = instances_.address;
    p.lights = lights_.address;
    p.pickResult = pickBuf_.address;
    p.cam = f.camera;
    p.prevCam = f.prevCamera;
    p.width = renderW_;
    p.height = renderH_;
    p.frameIndex = f.frameIndex;
    p.accumCount = accumCount_;
    p.spp = f.spp;
    p.maxBounces = f.maxBounces;
    p.flags = 2u /* accumulation */ | (f.fireflyClamp ? 8u : 0u);
    p.numLights = numLights_;
    p.envIntensity = 1.0f;
    p.pickX = f.pickX;
    p.pickY = f.pickY;
    std::memcpy(frameUbo_.mapped, &p, sizeof(p));
}

void VkRenderer::recordPick(VkCommandBuffer cmd, const VkFrameDesc& f)
{
    writeParams(f);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pick_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pick_.layout, 0, 1, &pickSet_, 0, nullptr);
    vkCmdDispatch(cmd, 1, 1, 1);
    VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
}

void VkRenderer::recordFrame(VkCommandBuffer cmd, const VkFrameDesc& f)
{
    const uint32_t rw = std::clamp(f.renderW ? f.renderW : displayW_, 1u, displayW_);
    const uint32_t rh = std::clamp(f.renderH ? f.renderH : displayH_, 1u, displayH_);
    if (rw != renderW_ || rh != renderH_) {
        accumCount_ = 0;
        historyValid_ = false;
    }
    if (!f.temporal) historyValid_ = false;
    renderW_ = rw;
    renderH_ = rh;
    bloomW_ = (renderW_ + 3) / 4;
    bloomH_ = (renderH_ + 3) / 4;
    if (f.resetAccumulation) accumCount_ = 0;
    writeParams(f);

    if (queries_) {
        vkCmdResetQueryPool(cmd, queries_, 0, kQueryCount);
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, queries_, 0);
    }
    // ---- path tracing (ray query) + sélection sous le curseur
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trace_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, trace_.layout, 0, 1, &traceSet_, 0, nullptr);
    vkCmdDispatch(cmd, groupsFor(renderW_, 8), groupsFor(renderH_, 8), 1);
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pick_.pipeline);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pick_.layout, 0, 1, &pickSet_, 0, nullptr);
    vkCmdDispatch(cmd, 1, 1, 1);
    VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_HOST_BIT,
                                 VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_HOST_READ_BIT);
    if (queries_) vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, queries_, 1);
    ++accumCount_;
    if (!f.runPost) {
        if (queries_) {
            vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 2);
            queriesWritten_ = true;
        }
        return;
    }

    // ---- images en mouvement : accumulation temporelle reprojetée
    uint64_t radiance = color_.address;
    uint32_t spp = accumCount_ * std::max(1u, f.spp);
    if (f.temporal) {
        TemporalPush tp{};
        tp.color = color_.address;
        tp.normal = normal_.address;
        tp.motion = motion_.address;
        tp.histIn = hist_[histIndex_].address;
        tp.histOut = hist_[histIndex_ ^ 1].address;
        tp.depthIn = histDepth_[histIndex_].address;
        tp.depthOut = histDepth_[histIndex_ ^ 1].address;
        tp.width = renderW_;
        tp.height = renderH_;
        tp.valid = historyValid_ ? 1u : 0u;
        tp.maxHistory = 10.0f;
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, temporal_.pipeline);
        vkCmdPushConstants(cmd, temporal_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(tp), &tp);
        vkCmdDispatch(cmd, groupsFor(renderW_, 8), groupsFor(renderH_, 8), 1);
        VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                     VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT);
        radiance = tp.histOut;
        histIndex_ ^= 1;
        spp = historyValid_ ? 6u : 1u;  // estimation du nombre d'échantillons effectifs
        historyValid_ = true;
    }

    // ---- débruitage à-trous : force décroissante avec le nombre d'échantillons accumulés
    uint64_t hdr = radiance;
    int iterations = 0;
    if (f.denoise) iterations = spp < 2 ? 5 : spp < 8 ? 4 : spp < 32 ? 3 : spp < 128 ? 2 : 0;
    if (iterations > 0) {
        const uint64_t ping[2] = {denoiseA_.address, denoiseB_.address};
        uint64_t src = radiance;
        for (int it = 0; it < iterations; ++it) {
            PostPush pp{};
            pp.src = src;
            pp.dst = ping[it & 1];
            pp.aux0 = albedo_.address;
            pp.aux1 = normal_.address;
            pp.srcW = renderW_;
            pp.srcH = renderH_;
            pp.p0 = 64.0f;                                           // normales
            pp.p1 = 0.02f;                                           // profondeur (relative)
            pp.p2 = spp < 4 ? 0.0f : 1.5f / std::sqrt(static_cast<float>(spp));  // luminance (ignorée si très bruité)
            pp.p3 = 0.12f;                                           // albédo
            pp.u0 = 1u << it;
            pp.u1 = (it == 0 ? 1u : 0u) | (it == iterations - 1 ? 2u : 0u);
            post(cmd, denoise_, pp, renderW_, renderH_, 8);
            src = pp.dst;
        }
        hdr = src;
    }

    // ---- bloom
    PostPush pb{};
    pb.src = hdr;
    pb.dst = bloomA_.address;
    pb.srcW = renderW_;
    pb.srcH = renderH_;
    pb.dstW = bloomW_;
    pb.dstH = bloomH_;
    pb.p0 = f.exposure;
    pb.p1 = f.bloomThreshold;
    post(cmd, bloomDown_, pb, bloomW_, bloomH_, 16);
    PostPush pbl{};
    pbl.src = bloomA_.address;
    pbl.dst = bloomB_.address;
    pbl.dstW = bloomW_;
    pbl.dstH = bloomH_;
    pbl.u0 = 1;
    post(cmd, bloomBlur_, pbl, bloomW_, bloomH_, 16);
    pbl.src = bloomB_.address;
    pbl.dst = bloomA_.address;
    pbl.u0 = 0;
    pbl.u1 = 1;
    post(cmd, bloomBlur_, pbl, bloomW_, bloomH_, 16);

    // ---- composition (tone mapping) en résolution de rendu
    PostPush pc{};
    pc.src = hdr;
    pc.aux0 = bloomA_.address;
    pc.dst = ldr_.address;
    pc.srcW = renderW_;
    pc.srcH = renderH_;
    pc.dstW = bloomW_;
    pc.dstH = bloomH_;
    pc.p0 = f.exposure;
    pc.p1 = f.bloomStrength;
    pc.p2 = f.vignette;
    post(cmd, composite_, pc, renderW_, renderH_, 16);

    // ---- AMD FSR 1 : EASU (suréchantillonnage) + RCAS (netteté)
    uint64_t finalLdr = ldr_.address;
    if (renderW_ != displayW_ || renderH_ != displayH_) {
        PostPush pe{};
        pe.src = ldr_.address;
        pe.dst = easuOut_.address;
        pe.srcW = renderW_;
        pe.srcH = renderH_;
        pe.dstW = displayW_;
        pe.dstH = displayH_;
        post(cmd, easu_, pe, displayW_, displayH_, 8);
        PostPush pr{};
        pr.src = easuOut_.address;
        pr.dst = rcasOut_.address;
        pr.srcW = displayW_;
        pr.srcH = displayH_;
        pr.dstW = displayW_;
        pr.dstH = displayH_;
        pr.p0 = f.sharpness;
        post(cmd, rcas_, pr, displayW_, displayH_, 8);
        finalLdr = rcasOut_.address;
    }

    // ---- octets de la swapchain
    PostPush pk{};
    pk.src = finalLdr;
    pk.dst = present_.address;
    pk.dstW = displayW_;
    pk.dstH = displayH_;
    pk.u0 = f.bgra ? 1u : 0u;
    pk.u1 = f.frameIndex;
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, pack_.pipeline);
    vkCmdPushConstants(cmd, pack_.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(PostPush), &pk);
    vkCmdDispatch(cmd, groupsFor(displayW_, 16), groupsFor(displayH_, 16), 1);
    VulkanContext::memoryBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
                                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
    if (queries_) {
        vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, queries_, 2);
        queriesWritten_ = true;
    }
}

}  // namespace crtx
