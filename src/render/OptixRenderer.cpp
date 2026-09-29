// CalculatoRTX - moteur de rendu OptiX : pipeline, SBT, BVH (GAS/IAS), Opacity Micromaps,
// débruiteur IA et sélection par rayon.
#include "OptixRenderer.h"

#include "Grille.h"
#include "Kernels.h"
#include "OptixCheck.h"

#include <optix_function_table_definition.h>
#include <optix_micromap.h>
#include <optix_stack_size.h>
#include <optix_stubs.h>

#include <algorithm>
#include <cstring>

namespace crtx {

namespace {

struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) SbtRecord {
    char header[OPTIX_SBT_RECORD_HEADER_SIZE];
};

void optixLogCallback(unsigned int level, const char* tag, const char* message, void*)
{
    std::fprintf(stderr, "[OptiX %u][%s] %s\n", level, tag, message);
}

template <typename T>
void uploadVector(CUdeviceptr& dst, const std::vector<T>& v)
{
    dst = 0;
    if (v.empty()) return;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dst), sizeof(T) * v.size()));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(dst), v.data(), sizeof(T) * v.size(), cudaMemcpyHostToDevice));
}

template <typename T>
void ensureCapacity(T*& ptr, size_t& cap, size_t count)
{
    if (count <= cap && ptr) return;
    cudaFree(ptr);
    cap = std::max<size_t>(count, 16);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * cap));
}

void freePtr(CUdeviceptr& p)
{
    if (p) cudaFree(reinterpret_cast<void*>(p));
    p = 0;
}

OptixImage2D image2D(const void* data, unsigned w, unsigned h)
{
    OptixImage2D img{};
    img.data = reinterpret_cast<CUdeviceptr>(data);
    img.width = w;
    img.height = h;
    img.rowStrideInBytes = w * sizeof(float4);
    img.pixelStrideInBytes = sizeof(float4);
    img.format = OPTIX_PIXEL_FORMAT_FLOAT4;
    return img;
}

}  // namespace

OptixRenderer::OptixRenderer() = default;

OptixRenderer::~OptixRenderer()
{
    cudaDeviceSynchronize();
    destroyDenoiser();
    freeFrameBuffers();
    for (Gas& g : gas_) freeGas(g);
    freePtr(iasBuffer_);
    freePtr(iasTemp_);
    freePtr(instanceBuffer_);
    freePtr(sbtBuffer_);
    cudaFree(dMaterials_);
    cudaFree(dGeometries_);
    cudaFree(dInstances_);
    cudaFree(dLights_);
    cudaFree(dParams_);
    cudaFreeHost(hParams_);
    cudaFree(dPick_);
    cudaFreeHost(hostPick_);
    if (envTex_) cudaDestroyTextureObject(envTex_);
    if (envArray_) cudaFreeArray(envArray_);
    if (evTrace0_) cudaEventDestroy(evTrace0_);
    if (evTrace1_) cudaEventDestroy(evTrace1_);
    if (evDenoise1_) cudaEventDestroy(evDenoise1_);
    if (pipeline_) optixPipelineDestroy(pipeline_);
    for (OptixProgramGroup g : {pgRaygen_, pgPick_, pgMiss_, pgHitOpaque_, pgHitCutout_})
        if (g) optixProgramGroupDestroy(g);
    if (module_) optixModuleDestroy(module_);
    if (context_) optixDeviceContextDestroy(context_);
}

void OptixRenderer::init(const std::vector<unsigned char>& ptx)
{
    createContext();
    createModuleAndPipeline(ptx);
    createSbt();
    createEnvironment();
    // deux blocs de paramètres (rendu + sélection) pour éviter toute réécriture en vol
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dParams_), 2 * sizeof(LaunchParams)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hParams_), 2 * sizeof(LaunchParams)));
    std::memset(hParams_, 0, 2 * sizeof(LaunchParams));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&dPick_), sizeof(int)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&hostPick_), sizeof(int)));
    *hostPick_ = -1;
    CUDA_CHECK(cudaEventCreate(&evTrace0_));
    CUDA_CHECK(cudaEventCreate(&evTrace1_));
    CUDA_CHECK(cudaEventCreate(&evDenoise1_));
}

void OptixRenderer::createContext()
{
    CUDA_CHECK(cudaFree(nullptr));  // initialise le contexte CUDA primaire
    OPTIX_CHECK(optixInit());
    OptixDeviceContextOptions opts{};
    opts.logCallbackFunction = &optixLogCallback;
    opts.logCallbackLevel = 2;  // erreurs + avertissements
    OPTIX_CHECK(optixDeviceContextCreate(nullptr, &opts, &context_));
    optixDeviceContextSetCacheEnabled(context_, 1);  // cache disque des modules compilés
}

void OptixRenderer::createModuleAndPipeline(const std::vector<unsigned char>& ptx)
{
    OptixModuleCompileOptions mco{};
    mco.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    mco.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    mco.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pco{};
    pco.usesMotionBlur = 0;
    pco.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pco.numPayloadValues = 0;    // style "hit object" (SER) : aucune charge utile
    pco.numAttributeValues = 2;  // barycentriques des triangles
    pco.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pco.pipelineLaunchParamsVariableName = "params";
#if OPTIX_VERSION >= 90100
    pco.pipelineLaunchParamsSizeInBytes = sizeof(LaunchParams);
#endif
    pco.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pco.allowOpacityMicromaps = 1;  // Opacity Micromaps (accélération matérielle Ada)

    OPTIX_CHECK_LOG(optixModuleCreate(context_, &mco, &pco, reinterpret_cast<const char*>(ptx.data()), ptx.size(),
                                      log, &logSize, &module_));

    OptixProgramGroupOptions pgo{};
    OptixProgramGroupDesc desc[5] = {};
    desc[0].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc[0].raygen.module = module_;
    desc[0].raygen.entryFunctionName = "__raygen__render";
    desc[1].kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    desc[1].raygen.module = module_;
    desc[1].raygen.entryFunctionName = "__raygen__pick";
    desc[2].kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    desc[2].miss.module = module_;
    desc[2].miss.entryFunctionName = "__miss__noop";
    desc[3].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc[3].hitgroup.moduleCH = module_;
    desc[3].hitgroup.entryFunctionNameCH = "__closesthit__noop";
    desc[4].kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    desc[4].hitgroup.moduleCH = module_;
    desc[4].hitgroup.entryFunctionNameCH = "__closesthit__noop";
    desc[4].hitgroup.moduleAH = module_;
    desc[4].hitgroup.entryFunctionNameAH = "__anyhit__cutout";
    OptixProgramGroup groups[5] = {};
    OPTIX_CHECK_LOG(optixProgramGroupCreate(context_, desc, 5, &pgo, log, &logSize, groups));
    pgRaygen_ = groups[0];
    pgPick_ = groups[1];
    pgMiss_ = groups[2];
    pgHitOpaque_ = groups[3];
    pgHitCutout_ = groups[4];

    OptixPipelineLinkOptions plo{};
    plo.maxTraceDepth = 1;
    OPTIX_CHECK_LOG(optixPipelineCreate(context_, &pco, &plo, groups, 5, log, &logSize, &pipeline_));

    OptixStackSizes ss{};
    for (OptixProgramGroup g : groups) OPTIX_CHECK(optixUtilAccumulateStackSizes(g, &ss, pipeline_));
    unsigned dcTrav = 0, dcState = 0, cont = 0;
    OPTIX_CHECK(optixUtilComputeStackSizes(&ss, 1, 0, 0, &dcTrav, &dcState, &cont));
    OPTIX_CHECK(optixPipelineSetStackSize(pipeline_, dcTrav, dcState, cont, 2));
}

void OptixRenderer::createSbt()
{
    SbtRecord rec[5];
    OPTIX_CHECK(optixSbtRecordPackHeader(pgRaygen_, &rec[0]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgPick_, &rec[1]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgMiss_, &rec[2]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgHitOpaque_, &rec[3]));
    OPTIX_CHECK(optixSbtRecordPackHeader(pgHitCutout_, &rec[4]));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&sbtBuffer_), sizeof(rec)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(sbtBuffer_), rec, sizeof(rec), cudaMemcpyHostToDevice));
    const CUdeviceptr base = sbtBuffer_;
    sbt_ = OptixShaderBindingTable{};
    sbt_.raygenRecord = base;
    sbt_.missRecordBase = base + 2 * sizeof(SbtRecord);
    sbt_.missRecordStrideInBytes = sizeof(SbtRecord);
    sbt_.missRecordCount = 1;
    sbt_.hitgroupRecordBase = base + 3 * sizeof(SbtRecord);
    sbt_.hitgroupRecordStrideInBytes = sizeof(SbtRecord);
    sbt_.hitgroupRecordCount = 2;
    sbtPick_ = sbt_;
    sbtPick_.raygenRecord = base + sizeof(SbtRecord);
}

void OptixRenderer::createEnvironment()
{
    const int w = 1024, h = 512;
    float4* tmp = nullptr;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&tmp), sizeof(float4) * w * h));
    kernels::bakeEnvironment(tmp, w, h, nullptr);
    const cudaChannelFormatDesc fmt = cudaCreateChannelDesc<float4>();
    CUDA_CHECK(cudaMallocArray(&envArray_, &fmt, w, h));
    CUDA_CHECK(cudaMemcpy2DToArray(envArray_, 0, 0, tmp, w * sizeof(float4), w * sizeof(float4), h,
                                   cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaFree(tmp));
    cudaResourceDesc res{};
    res.resType = cudaResourceTypeArray;
    res.res.array.array = envArray_;
    cudaTextureDesc td{};
    td.addressMode[0] = cudaAddressModeWrap;
    td.addressMode[1] = cudaAddressModeClamp;
    td.filterMode = cudaFilterModeLinear;  // filtrage bilinéaire par les unités de texture
    td.readMode = cudaReadModeElementType;
    td.normalizedCoords = 1;
    CUDA_CHECK(cudaCreateTextureObject(&envTex_, &res, &td, nullptr));
}

// ---------------------------------------------------------------- Opacity Micromaps
void OptixRenderer::buildOpacityMicromap(Gas& gas, const Mesh& mesh, cudaStream_t stream)
{
    const unsigned level = grille::kSubdivisionLevel;
    const unsigned microTris = 1u << (2 * level);          // 4^level
    const unsigned bytesPerTri = microTris * 2 / 8;        // 2 bits par micro-triangle (4 états)
    const size_t numTris = mesh.indices.size();
    std::vector<unsigned char> data(numTris * bytesPerTri, 0);
    std::vector<OptixOpacityMicromapDesc> descs(numTris);
    size_t unknown = 0;

    for (size_t t = 0; t < numTris; ++t) {
        const uint3 tri = mesh.indices[t];
        const float2 uv0 = mesh.uvs[tri.x], uv1 = mesh.uvs[tri.y], uv2 = mesh.uvs[tri.z];
        auto uvAt = [&](float2 b) { return uv0 * (1.0f - b.x - b.y) + uv1 * b.x + uv2 * b.y; };
        auto toWorld = [](float2 uv) { return make_float2(uv.x * grille::kWidth, uv.y * grille::kHeight); };
        unsigned char* dst = data.data() + t * bytesPerTri;
        for (unsigned m = 0; m < microTris; ++m) {
            float2 b0, b1, b2;
            optixMicromapIndexToBaseBarycentrics(m, level, b0, b1, b2);
            const float2 q0 = uvAt(b0), q1 = uvAt(b1), q2 = uvAt(b2);
            const float2 qc = (q0 + q1 + q2) * (1.0f / 3.0f);
            // rayon circonscrit du micro-triangle (espace monde de la plaque)
            float r = 0.0f;
            for (float2 q : {q0, q1, q2}) {
                const float2 d = toWorld(q) - toWorld(qc);
                r = std::max(r, std::sqrt(d.x * d.x + d.y * d.y));
            }
            const float s = grille::sdf(qc);  // fonction 1-lipschitzienne => classement exact
            unsigned state;
            if (s > r) state = OPTIX_OPACITY_MICROMAP_STATE_OPAQUE;
            else if (s < -r) state = OPTIX_OPACITY_MICROMAP_STATE_TRANSPARENT;
            else {
                state = s >= 0.0f ? OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_OPAQUE
                                  : OPTIX_OPACITY_MICROMAP_STATE_UNKNOWN_TRANSPARENT;
                ++unknown;
            }
            dst[m / 4] |= static_cast<unsigned char>(state << (2 * (m % 4)));
        }
        descs[t].byteOffset = static_cast<unsigned>(t * bytesPerTri);
        descs[t].subdivisionLevel = static_cast<unsigned short>(level);
        descs[t].format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
    }

    CUdeviceptr dData = 0, dDescs = 0;
    uploadVector(dData, data);
    uploadVector(dDescs, descs);

    OptixOpacityMicromapHistogramEntry hist{};
    hist.count = static_cast<unsigned>(numTris);
    hist.subdivisionLevel = level;
    hist.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;

    OptixOpacityMicromapArrayBuildInput bi{};
    bi.flags = OPTIX_OPACITY_MICROMAP_FLAG_PREFER_FAST_TRACE;
    bi.inputBuffer = dData;
    bi.perMicromapDescBuffer = dDescs;
    bi.perMicromapDescStrideInBytes = sizeof(OptixOpacityMicromapDesc);
    bi.numMicromapHistogramEntries = 1;
    bi.micromapHistogramEntries = &hist;

    OptixMicromapBufferSizes sizes{};
    OPTIX_CHECK(optixOpacityMicromapArrayComputeMemoryUsage(context_, &bi, &sizes));
    OptixMicromapBuffers bufs{};
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&bufs.output), sizes.outputSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&bufs.temp), sizes.tempSizeInBytes));
    bufs.outputSizeInBytes = sizes.outputSizeInBytes;
    bufs.tempSizeInBytes = sizes.tempSizeInBytes;
    OPTIX_CHECK(optixOpacityMicromapArrayBuild(context_, stream, &bi, &bufs));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    freePtr(bufs.temp);
    freePtr(dData);
    freePtr(dDescs);
    gas.ommArray = bufs.output;
    CRTX_LOG("Opacity Micromaps : %zu triangles x %u micro-triangles, %.1f %% indéterminés (any-hit)", numTris,
             microTris, 100.0 * static_cast<double>(unknown) / (static_cast<double>(numTris) * microTris));
}

// ---------------------------------------------------------------- BVH des maillages (GAS)
void OptixRenderer::freeGas(Gas& g)
{
    freePtr(g.buffer);
    freePtr(g.positions);
    freePtr(g.normals);
    freePtr(g.uvs);
    freePtr(g.indices);
    freePtr(g.ommArray);
    g.handle = 0;
    g.triangles = 0;
}

void OptixRenderer::buildGas(Gas& gas, const SceneMesh& sm, cudaStream_t stream)
{
    freeGas(gas);
    const Mesh& mesh = sm.mesh;
    gas.alpha = sm.alphaCutout;
    if (mesh.empty()) return;
    uploadVector(gas.positions, mesh.positions);
    if (!mesh.flat) uploadVector(gas.normals, mesh.normals);
    uploadVector(gas.uvs, mesh.uvs);
    uploadVector(gas.indices, mesh.indices);
    gas.triangles = mesh.indices.size();

    if (gas.alpha) buildOpacityMicromap(gas, mesh, stream);

    OptixBuildInput bi{};
    bi.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    OptixBuildInputTriangleArray& ta = bi.triangleArray;
    ta.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    ta.vertexStrideInBytes = sizeof(float3);
    ta.numVertices = static_cast<unsigned>(mesh.positions.size());
    ta.vertexBuffers = &gas.positions;
    ta.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    ta.indexStrideInBytes = sizeof(uint3);
    ta.numIndexTriplets = static_cast<unsigned>(mesh.indices.size());
    ta.indexBuffer = gas.indices;
    const unsigned flags = gas.alpha ? OPTIX_GEOMETRY_FLAG_NONE : OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT;
    ta.flags = &flags;
    ta.numSbtRecords = 1;
    OptixOpacityMicromapUsageCount usage{};
    if (gas.alpha) {
        usage.count = static_cast<unsigned>(mesh.indices.size());
        usage.subdivisionLevel = grille::kSubdivisionLevel;
        usage.format = OPTIX_OPACITY_MICROMAP_FORMAT_4_STATE;
        ta.opacityMicromap.indexingMode = OPTIX_OPACITY_MICROMAP_ARRAY_INDEXING_MODE_LINEAR;
        ta.opacityMicromap.opacityMicromapArray = gas.ommArray;
        ta.opacityMicromap.numMicromapUsageCounts = 1;
        ta.opacityMicromap.micromapUsageCounts = &usage;
    }

    // Compaction pour les gros maillages statiques ; les petits maillages de texte (reconstruits
    // à chaque frappe) sont construits sans synchronisation CPU.
    const bool compact = mesh.indices.size() > 4096;
    OptixAccelBuildOptions ao{};
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | (compact ? static_cast<unsigned>(OPTIX_BUILD_FLAG_ALLOW_COMPACTION) : 0u);
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &ao, &bi, 1, &sizes));

    CUdeviceptr temp = 0, out = 0, compactedSize = 0;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&temp), sizes.tempSizeInBytes));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&out), sizes.outputSizeInBytes));
    OptixAccelEmitDesc emit{};
    if (compact) {
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&compactedSize), sizeof(size_t)));
        emit.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
        emit.result = compactedSize;
    }
    OPTIX_CHECK(optixAccelBuild(context_, stream, &ao, &bi, 1, temp, sizes.tempSizeInBytes, out,
                                sizes.outputSizeInBytes, &gas.handle, compact ? &emit : nullptr, compact ? 1u : 0u));
    if (compact) {
        size_t compacted = 0;
        CUDA_CHECK(cudaMemcpyAsync(&compacted, reinterpret_cast<void*>(compactedSize), sizeof(size_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        if (compacted > 0 && compacted < sizes.outputSizeInBytes) {
            CUdeviceptr small = 0;
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&small), compacted));
            OPTIX_CHECK(optixAccelCompact(context_, stream, gas.handle, small, compacted, &gas.handle));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            freePtr(out);
            out = small;
        }
    }
    gas.buffer = out;
    freePtr(temp);  // cudaFree synchronise implicitement : la construction est terminée
    freePtr(compactedSize);
}

// ---------------------------------------------------------------- BVH de haut niveau (IAS)
void OptixRenderer::buildIas(const CalculatorScene& scene, cudaStream_t stream)
{
    const auto& instances = scene.instances();
    if (prevTransforms_.size() != instances.size()) {
        prevTransforms_.resize(instances.size());
        for (size_t i = 0; i < instances.size(); ++i) prevTransforms_[i] = instances[i].transform;
    }
    std::vector<OptixInstance> oi;
    std::vector<InstanceData> idata;
    oi.reserve(instances.size());
    idata.reserve(instances.size());
    for (size_t i = 0; i < instances.size(); ++i) {
        const SceneInstance& si = instances[i];
        if (si.mesh < 0 || si.mesh >= static_cast<int>(gas_.size())) continue;
        const Gas& g = gas_[si.mesh];
        if (!g.handle) continue;  // maillage vide (ex. texte vide)
        OptixInstance x{};
        std::memcpy(x.transform, si.transform.m, sizeof(x.transform));
        x.instanceId = static_cast<unsigned>(i);
        x.sbtOffset = g.alpha ? 1u : 0u;
        x.visibilityMask = si.mask;
        x.flags = OPTIX_INSTANCE_FLAG_NONE;
        x.traversableHandle = g.handle;
        oi.push_back(x);
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

    const size_t bytes = sizeof(OptixInstance) * std::max<size_t>(oi.size(), 1);
    if (bytes > instanceBufferSize_) {
        freePtr(instanceBuffer_);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&instanceBuffer_), bytes * 2));
        instanceBufferSize_ = bytes * 2;
    }
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(instanceBuffer_), oi.data(), sizeof(OptixInstance) * oi.size(),
                               cudaMemcpyHostToDevice, stream));
    ensureCapacity(dInstances_, instanceCap_, idata.size());
    CUDA_CHECK(cudaMemcpyAsync(dInstances_, idata.data(), sizeof(InstanceData) * idata.size(),
                               cudaMemcpyHostToDevice, stream));

    OptixBuildInput bi{};
    bi.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    bi.instanceArray.instances = instanceBuffer_;
    bi.instanceArray.numInstances = static_cast<unsigned>(oi.size());
    OptixAccelBuildOptions ao{};
    ao.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_BUILD;
    ao.operation = OPTIX_BUILD_OPERATION_BUILD;
    OptixAccelBufferSizes sizes{};
    OPTIX_CHECK(optixAccelComputeMemoryUsage(context_, &ao, &bi, 1, &sizes));
    if (sizes.outputSizeInBytes > iasBufferSize_) {
        freePtr(iasBuffer_);
        iasBufferSize_ = sizes.outputSizeInBytes * 2;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iasBuffer_), iasBufferSize_));
    }
    if (sizes.tempSizeInBytes > iasTempSize_) {
        freePtr(iasTemp_);
        iasTempSize_ = sizes.tempSizeInBytes * 2;
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&iasTemp_), iasTempSize_));
    }
    OPTIX_CHECK(optixAccelBuild(context_, stream, &ao, &bi, 1, iasTemp_, iasTempSize_, iasBuffer_, iasBufferSize_,
                                &ias_, nullptr, 0));
}

void OptixRenderer::syncScene(CalculatorScene& scene, cudaStream_t stream)
{
    auto& meshes = scene.meshes();
    if (gas_.size() < meshes.size()) gas_.resize(meshes.size());
    bool geometryChanged = false;
    for (size_t i = 0; i < meshes.size(); ++i) {
        if (!meshes[i].dirty) continue;
        buildGas(gas_[i], meshes[i], stream);
        meshes[i].dirty = false;
        geometryChanged = true;
    }
    if (geometryChanged || !dGeometries_) {
        std::vector<GeometryData> geo(gas_.size());
        for (size_t i = 0; i < gas_.size(); ++i) {
            geo[i].positions = reinterpret_cast<const float3*>(gas_[i].positions);
            geo[i].normals = reinterpret_cast<const float3*>(gas_[i].normals);
            geo[i].uvs = reinterpret_cast<const float2*>(gas_[i].uvs);
            geo[i].indices = reinterpret_cast<const uint3*>(gas_[i].indices);
        }
        ensureCapacity(dGeometries_, geometryCap_, geo.size());
        CUDA_CHECK(cudaMemcpyAsync(dGeometries_, geo.data(), sizeof(GeometryData) * geo.size(),
                                   cudaMemcpyHostToDevice, stream));
        accumCount_ = 0;
    }
    const auto& mats = scene.materials();
    ensureCapacity(dMaterials_, materialCap_, mats.size());
    CUDA_CHECK(cudaMemcpyAsync(dMaterials_, mats.data(), sizeof(Material) * mats.size(), cudaMemcpyHostToDevice, stream));
    const auto& lights = scene.lights();
    ensureCapacity(dLights_, lightCap_, lights.size());
    if (!lights.empty())
        CUDA_CHECK(cudaMemcpyAsync(dLights_, lights.data(), sizeof(RectLight) * lights.size(), cudaMemcpyHostToDevice,
                                   stream));
    numLights_ = static_cast<unsigned>(lights.size());
    buildIas(scene, stream);
    // Les copies ci-dessus partent de mémoire pageable (copie synchrone côté hôte),
    // les vecteurs temporaires peuvent donc être libérés sans risque.
}

// ---------------------------------------------------------------- images & débruiteur
void OptixRenderer::freeFrameBuffers()
{
    cudaFree(frameBlock_);
    cudaFree(depth_);
    cudaFree(motion_);
    frameBlock_ = nullptr;
    depth_ = nullptr;
    motion_ = nullptr;
    color_ = albedo_ = normal_ = accum_ = denoised_ = nullptr;
}

void OptixRenderer::resize(unsigned width, unsigned height)
{
    if (width == width_ && height == height_ && frameBlock_) return;
    CUDA_CHECK(cudaDeviceSynchronize());
    destroyDenoiser();
    freeFrameBuffers();
    width_ = std::max(1u, width);
    height_ = std::max(1u, height);
    const size_t n = static_cast<size_t>(width_) * height_;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&frameBlock_), sizeof(float4) * n * 5));
    CUDA_CHECK(cudaMemset(frameBlock_, 0, sizeof(float4) * n * 5));
    color_ = frameBlock_;
    albedo_ = frameBlock_ + n;
    normal_ = frameBlock_ + 2 * n;
    denoised_ = frameBlock_ + 3 * n;
    accum_ = frameBlock_ + 4 * n;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&depth_), sizeof(float) * n));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&motion_), sizeof(float2) * n));
    accumCount_ = 0;
    l2Configured_ = false;
    setupDenoiser();
}

void OptixRenderer::setupDenoiser()
{
    OptixDenoiserOptions opt{};
    opt.guideAlbedo = 1;
    opt.guideNormal = 1;
    opt.denoiseAlpha = OPTIX_DENOISER_ALPHA_MODE_COPY;
    OPTIX_CHECK(optixDenoiserCreate(context_, OPTIX_DENOISER_MODEL_KIND_AOV, &opt, &denoiser_));
    OptixDenoiserSizes ds{};
    OPTIX_CHECK(optixDenoiserComputeMemoryResources(denoiser_, width_, height_, &ds));
    denoiserStateSize_ = ds.stateSizeInBytes;
    denoiserScratchSize_ = std::max(ds.withoutOverlapScratchSizeInBytes, ds.computeIntensitySizeInBytes);
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&denoiserState_), denoiserStateSize_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&denoiserScratch_), denoiserScratchSize_));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&denoiserIntensity_), sizeof(float)));
    OPTIX_CHECK(optixDenoiserSetup(denoiser_, nullptr, width_, height_, denoiserState_, denoiserStateSize_,
                                   denoiserScratch_, denoiserScratchSize_));
}

void OptixRenderer::destroyDenoiser()
{
    if (denoiser_) optixDenoiserDestroy(denoiser_);
    denoiser_ = nullptr;
    freePtr(denoiserState_);
    freePtr(denoiserScratch_);
    freePtr(denoiserIntensity_);
}

// Grand cache L2 d'Ada (48 Mo sur AD104) : fenêtre "persistante" sur les images
// produites par le path tracer puis relues par le débruiteur.
void OptixRenderer::applyL2Persistence(cudaStream_t stream)
{
    if (l2Configured_) return;
    l2Configured_ = true;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    int maxPersist = 0, maxWindow = 0;
    cudaDeviceGetAttribute(&maxPersist, cudaDevAttrMaxPersistingL2CacheSize, dev);
    cudaDeviceGetAttribute(&maxWindow, cudaDevAttrMaxAccessPolicyWindowSize, dev);
    if (maxPersist <= 0 || maxWindow <= 0) return;
    const size_t bytes = sizeof(float4) * static_cast<size_t>(width_) * height_ * 3;  // color+albedo+normal
    const size_t window = std::min(bytes, static_cast<size_t>(maxWindow));
    const size_t persist = std::min(static_cast<size_t>(maxPersist), window);
    if (cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, persist) != cudaSuccess) {
        cudaGetLastError();
        return;
    }
    cudaStreamAttrValue attr{};
    attr.accessPolicyWindow.base_ptr = frameBlock_;
    attr.accessPolicyWindow.num_bytes = window;
    attr.accessPolicyWindow.hitRatio = std::min(1.0f, static_cast<float>(persist) / static_cast<float>(window));
    attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    if (cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr) != cudaSuccess) cudaGetLastError();
    else CRTX_LOG("Cache L2 : fenêtre persistante de %.1f Mo (hitRatio %.2f)", window / 1048576.0,
                  attr.accessPolicyWindow.hitRatio);
}

// ---------------------------------------------------------------- lancement
const float4* OptixRenderer::render(const FrameInput& in, const RenderSettings& rs, cudaStream_t stream)
{
    applyL2Persistence(stream);
    if (timingPending_) {  // chronos de l'image précédente (déjà terminée)
        cudaEventElapsedTime(&traceMs_, evTrace0_, evTrace1_);
        if (rs.denoise) cudaEventElapsedTime(&denoiseMs_, evTrace1_, evDenoise1_);
        else denoiseMs_ = 0.0f;
    }

    const bool accumulate = !in.dlss;
    if (in.resetAccumulation || !accumulate) accumCount_ = 0;

    LaunchParams& p = hParams_[0];
    p.color = color_;
    p.albedo = albedo_;
    p.normal = normal_;
    p.depth = depth_;
    p.motion = motion_;
    p.accum = accum_;
    p.size = make_uint2(width_, height_);
    p.frameIndex = in.frameIndex;
    p.accumCount = accumCount_;
    p.spp = rs.spp;
    p.maxBounces = rs.maxBounces;
    p.flags = (rs.ser ? kFlagSER : 0u) | (accumulate ? kFlagAccumulate : 0u) | (in.dlss ? kFlagDLSS : 0u) |
              (rs.fireflyClamp ? kFlagFireflyClamp : 0u);
    p.jitter = in.jitter;
    p.cam = in.camera;
    p.prevCam = in.prevCamera;
    p.handle = ias_;
    p.materials = dMaterials_;
    p.geometries = dGeometries_;
    p.instances = dInstances_;
    p.lights = dLights_;
    p.numLights = numLights_;
    p.envTex = envTex_;
    p.envIntensity = 1.0f;
    p.pickPixel = make_float2(0, 0);
    p.pickResult = dPick_;
    p.time = in.time;

    CUDA_CHECK(cudaMemcpyAsync(dParams_, &p, sizeof(LaunchParams), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaEventRecord(evTrace0_, stream));
    OPTIX_CHECK(optixLaunch(pipeline_, stream, reinterpret_cast<CUdeviceptr>(dParams_), sizeof(LaunchParams), &sbt_,
                            width_, height_, 1));
    CUDA_CHECK(cudaEventRecord(evTrace1_, stream));
    if (accumulate) ++accumCount_;

    const float4* result = color_;
    if (rs.denoise && denoiser_) {
        const OptixImage2D colorImg = image2D(color_, width_, height_);
        OPTIX_CHECK(optixDenoiserComputeIntensity(denoiser_, stream, &colorImg, denoiserIntensity_, denoiserScratch_,
                                                  denoiserScratchSize_));
        OptixDenoiserParams dp{};
        dp.hdrIntensity = denoiserIntensity_;
        dp.blendFactor = 0.0f;
        OptixDenoiserGuideLayer guide{};
        guide.albedo = image2D(albedo_, width_, height_);
        guide.normal = image2D(normal_, width_, height_);
        OptixDenoiserLayer layer{};
        layer.input = colorImg;
        layer.output = image2D(denoised_, width_, height_);
        OPTIX_CHECK(optixDenoiserInvoke(denoiser_, stream, &dp, denoiserState_, denoiserStateSize_, &guide, &layer, 1,
                                        0, 0, denoiserScratch_, denoiserScratchSize_));
        result = denoised_;
    }
    CUDA_CHECK(cudaEventRecord(evDenoise1_, stream));
    timingPending_ = true;
    return result;
}

void OptixRenderer::pick(const CameraData& cam, float2 pixel, cudaStream_t stream)
{
    LaunchParams& p = hParams_[1];
    p = hParams_[0];
    p.cam = cam;
    p.pickPixel = pixel;
    p.pickResult = dPick_;
    p.handle = ias_;
    p.instances = dInstances_;
    p.size = make_uint2(width_, height_);
    CUDA_CHECK(cudaMemcpyAsync(dParams_ + 1, &p, sizeof(LaunchParams), cudaMemcpyHostToDevice, stream));
    OPTIX_CHECK(optixLaunch(pipeline_, stream, reinterpret_cast<CUdeviceptr>(dParams_ + 1), sizeof(LaunchParams),
                            &sbtPick_, 1, 1, 1));
    CUDA_CHECK(cudaMemcpyAsync(hostPick_, dPick_, sizeof(int), cudaMemcpyDeviceToHost, stream));
}

}  // namespace crtx
