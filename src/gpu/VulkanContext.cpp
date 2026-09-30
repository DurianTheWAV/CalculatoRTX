// CalculatoRTX - implémentation du contexte Vulkan (et, avec CUDA, de l'interopérabilité).
#include "VulkanContext.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <type_traits>

namespace crtx {

namespace {

#if defined(_WIN32)
constexpr VkExternalMemoryHandleTypeFlagBits kMemHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kSemHandleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT;
const char* kExtMemPlatform = VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME;
const char* kExtSemPlatform = VK_KHR_EXTERNAL_SEMAPHORE_WIN32_EXTENSION_NAME;
#else
constexpr VkExternalMemoryHandleTypeFlagBits kMemHandleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT;
constexpr VkExternalSemaphoreHandleTypeFlagBits kSemHandleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;
const char* kExtMemPlatform = VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME;
const char* kExtSemPlatform = VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME;
#endif

VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                             VkDebugUtilsMessageTypeFlagsEXT, const VkDebugUtilsMessengerCallbackDataEXT* data,
                                             void*)
{
    if (severity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
        std::fprintf(stderr, "[Vulkan] %s\n", data->pMessage);
    return VK_FALSE;
}

std::vector<VkExtensionProperties> deviceExtensions(VkPhysicalDevice d)
{
    uint32_t ec = 0;
    vkEnumerateDeviceExtensionProperties(d, nullptr, &ec, nullptr);
    std::vector<VkExtensionProperties> available(ec);
    vkEnumerateDeviceExtensionProperties(d, nullptr, &ec, available.data());
    return available;
}

bool hasExtension(const std::vector<VkExtensionProperties>& list, const std::string& name)
{
    return std::any_of(list.begin(), list.end(), [&](const VkExtensionProperties& e) { return name == e.extensionName; });
}

const char* kRtExtensions[] = {VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_RAY_QUERY_EXTENSION_NAME,
                               VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};

}  // namespace

VulkanContext::~VulkanContext()
{
    if (device_) vkDeviceWaitIdle(device_);
    destroySwapchain();
#if defined(CRTX_WITH_CUDA)
    if (cudaTimeline_) cudaDestroyExternalSemaphore(cudaTimeline_);
#endif
    if (timeline_) vkDestroySemaphore(device_, timeline_, nullptr);
    if (imageAvailable_) vkDestroySemaphore(device_, imageAvailable_, nullptr);
    if (commandPool_) vkDestroyCommandPool(device_, commandPool_, nullptr);
    if (device_) vkDestroyDevice(device_, nullptr);
    if (surface_) vkDestroySurfaceKHR(instance_, surface_, nullptr);
    if (messenger_) {
        auto fn = reinterpret_cast<PFN_vkDestroyDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkDestroyDebugUtilsMessengerEXT"));
        if (fn) fn(instance_, messenger_, nullptr);
    }
    if (instance_) vkDestroyInstance(instance_, nullptr);
}

void VulkanContext::createInstance(const std::vector<std::string>& extraInstanceExtensions, bool validation,
                                   bool headless)
{
    headless_ = headless;
    uint32_t count = 0;
    vkEnumerateInstanceExtensionProperties(nullptr, &count, nullptr);
    std::vector<VkExtensionProperties> available(count);
    vkEnumerateInstanceExtensionProperties(nullptr, &count, available.data());

    std::set<std::string> wanted;
    if (!headless) {
        uint32_t glfwCount = 0;
        const char** glfwExts = glfwGetRequiredInstanceExtensions(&glfwCount);
        if (!glfwExts) throwError("GLFW : Vulkan n'est pas disponible pour la présentation", __FILE__, __LINE__);
        for (uint32_t i = 0; i < glfwCount; ++i) wanted.insert(glfwExts[i]);
    }
    wanted.insert(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);
    wanted.insert(VK_KHR_EXTERNAL_MEMORY_CAPABILITIES_EXTENSION_NAME);
    wanted.insert(VK_KHR_EXTERNAL_SEMAPHORE_CAPABILITIES_EXTENSION_NAME);
    for (const std::string& e : extraInstanceExtensions) wanted.insert(e);

    std::vector<const char*> layers;
    if (validation) {
        uint32_t lc = 0;
        vkEnumerateInstanceLayerProperties(&lc, nullptr);
        std::vector<VkLayerProperties> lp(lc);
        vkEnumerateInstanceLayerProperties(&lc, lp.data());
        const bool hasLayer = std::any_of(lp.begin(), lp.end(), [](const VkLayerProperties& l) {
            return std::strcmp(l.layerName, "VK_LAYER_KHRONOS_validation") == 0;
        });
        if (hasLayer) {
            layers.push_back("VK_LAYER_KHRONOS_validation");
            wanted.insert(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
        } else {
            CRTX_LOG("Couche de validation Vulkan absente (ignorée)");
            validation = false;
        }
    }
    std::vector<std::string> enabled;
    for (const std::string& e : wanted) {
        if (hasExtension(available, e)) enabled.push_back(e);
        else CRTX_LOG("Extension d'instance Vulkan indisponible : %s", e.c_str());
    }
    std::vector<const char*> names;
    for (const std::string& e : enabled) names.push_back(e.c_str());

    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "CalculatoRTX";
    app.applicationVersion = VK_MAKE_VERSION(1, 1, 0);
    app.pEngineName = "CalculatoRTX";
    app.engineVersion = VK_MAKE_VERSION(1, 1, 0);
    app.apiVersion = VK_API_VERSION_1_3;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    ci.enabledExtensionCount = static_cast<uint32_t>(names.size());
    ci.ppEnabledExtensionNames = names.data();
    ci.enabledLayerCount = static_cast<uint32_t>(layers.size());
    ci.ppEnabledLayerNames = layers.data();
    VK_CHECK(vkCreateInstance(&ci, nullptr, &instance_));

    if (validation) {
        VkDebugUtilsMessengerCreateInfoEXT mi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
        mi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
        mi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                         VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT;
        mi.pfnUserCallback = &debugCallback;
        auto fn = reinterpret_cast<PFN_vkCreateDebugUtilsMessengerEXT>(
            vkGetInstanceProcAddr(instance_, "vkCreateDebugUtilsMessengerEXT"));
        if (fn) fn(instance_, &mi, nullptr, &messenger_);
    }
}

void VulkanContext::setPhysical(VkPhysicalDevice d)
{
    physical_ = d;
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(physical_, &p);
    deviceName_ = p.deviceName;
    vendorId_ = p.vendorID;
    discrete_ = p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    limits_ = p.limits;
    CRTX_LOG("Vulkan : %s (API %u.%u.%u)", p.deviceName, VK_VERSION_MAJOR(p.apiVersion), VK_VERSION_MINOR(p.apiVersion),
             VK_VERSION_PATCH(p.apiVersion));
    if (p.apiVersion < VK_API_VERSION_1_2) throwError("Vulkan 1.2 minimum requis", __FILE__, __LINE__);
}

void VulkanContext::pickPhysicalDeviceByUuid(const unsigned char uuid[16])
{
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
    VkPhysicalDevice fallback = VK_NULL_HANDLE;
    for (VkPhysicalDevice d : devices) {
        VkPhysicalDeviceIDProperties idp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &idp;
        vkGetPhysicalDeviceProperties2(d, &p2);
        if (std::memcmp(idp.deviceUUID, uuid, VK_UUID_SIZE) == 0) {
            setPhysical(d);
            return;
        }
        if (!fallback && p2.properties.vendorID == 0x10DE) fallback = d;
    }
    if (!fallback) throwError("Aucun GPU Vulkan correspondant au périphérique CUDA", __FILE__, __LINE__);
    CRTX_LOG("Attention : UUID CUDA/Vulkan différents, utilisation du premier GPU NVIDIA");
    setPhysical(fallback);
}

bool VulkanContext::deviceSupports(VkPhysicalDevice d, const DeviceRequest& req, std::string* why)
{
    VkPhysicalDeviceProperties p{};
    vkGetPhysicalDeviceProperties(d, &p);
    auto fail = [&](const std::string& w) {
        if (why) *why = w;
        return false;
    };
    if (p.apiVersion < VK_API_VERSION_1_2) return fail("Vulkan 1.2 requis");
    const auto ext = deviceExtensions(d);
    if (req.rayTracing) {
        for (const char* e : kRtExtensions)
            if (!hasExtension(ext, e)) return fail(std::string("extension ") + e + " absente (ray tracing matériel)");
    }
    VkPhysicalDeviceAccelerationStructureFeaturesKHR as{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    VkPhysicalDeviceRayQueryFeaturesKHR rq{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    if (req.rayTracing) {
        f12.pNext = &as;
        as.pNext = &rq;
    }
    vkGetPhysicalDeviceFeatures2(d, &f2);
    if (!f12.timelineSemaphore) return fail("sémaphores timeline absents");
    if (req.rayTracing) {
        if (!as.accelerationStructure || !rq.rayQuery) return fail("ray query non supporté");
        if (!f12.bufferDeviceAddress || !f12.scalarBlockLayout) return fail("bufferDeviceAddress / scalarBlockLayout absents");
    }
    if (req.float64 && (!f2.features.shaderFloat64 || !f2.features.shaderInt64))
        return fail("shaderFloat64 / shaderInt64 absents (calcul double précision)");
    return true;
}

void VulkanContext::pickPhysicalDevice(const DeviceRequest& req, int preferredIndex)
{
    uint32_t count = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, nullptr));
    std::vector<VkPhysicalDevice> devices(count);
    VK_CHECK(vkEnumeratePhysicalDevices(instance_, &count, devices.data()));
    if (count == 0) throwError("Aucun GPU Vulkan détecté", __FILE__, __LINE__);
    if (preferredIndex >= 0) {
        if (preferredIndex >= static_cast<int>(count)) throwError("Index de GPU Vulkan invalide", __FILE__, __LINE__);
        std::string why;
        if (!deviceSupports(devices[preferredIndex], req, &why)) throwError("GPU demandé incompatible : " + why, __FILE__, __LINE__);
        setPhysical(devices[preferredIndex]);
        return;
    }
    VkPhysicalDevice best = VK_NULL_HANDLE;
    int bestScore = -1;
    std::string reasons;
    for (VkPhysicalDevice d : devices) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        std::string why;
        if (!deviceSupports(d, req, &why)) {
            reasons += std::string("\n  - ") + p.deviceName + " : " + why;
            continue;
        }
        int score = 1;
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) score = 3;
        else if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) score = 2;
        if (score > bestScore) {
            bestScore = score;
            best = d;
        }
    }
    if (!best) throwError("Aucun GPU compatible (ray tracing Vulkan + FP64) :" + reasons, __FILE__, __LINE__);
    setPhysical(best);
}

void VulkanContext::createDevice(GLFWwindow* window, const DeviceRequest& req)
{
    if (window && !headless_) VK_CHECK(glfwCreateWindowSurface(instance_, window, nullptr, &surface_));

    uint32_t qc = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qc, nullptr);
    std::vector<VkQueueFamilyProperties> qf(qc);
    vkGetPhysicalDeviceQueueFamilyProperties(physical_, &qc, qf.data());
    bool found = false;
    for (uint32_t i = 0; i < qc; ++i) {
        VkBool32 present = VK_TRUE;
        if (surface_) vkGetPhysicalDeviceSurfaceSupportKHR(physical_, i, surface_, &present);
        const VkQueueFlags need = VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT;
        if ((qf[i].queueFlags & need) == need && present) {
            queueFamily_ = i;
            found = true;
            break;
        }
    }
    if (!found) throwError("Aucune file Vulkan graphique+calcul+présentation", __FILE__, __LINE__);
    timestampBits_.resize(qc);
    for (uint32_t i = 0; i < qc; ++i) timestampBits_[i] = qf[i].timestampValidBits;
    // File de calcul asynchrone (moteur de calcul de la calculatrice, indépendant du rendu)
    bool asyncCompute = false;
    for (uint32_t i = 0; i < qc; ++i) {
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT)) {
            computeFamily_ = i;
            asyncCompute = true;
            break;
        }
    }

    const auto available = deviceExtensions(physical_);
    std::set<std::string> wanted = {VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME, VK_KHR_DEDICATED_ALLOCATION_EXTENSION_NAME};
    if (surface_) wanted.insert(VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    if (req.cudaInterop) {
        wanted.insert(VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME);
        wanted.insert(VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME);
        wanted.insert(kExtMemPlatform);
        wanted.insert(kExtSemPlatform);
    }
    if (req.rayTracing)
        for (const char* e : kRtExtensions) wanted.insert(e);
    for (const std::string& e : req.extraExtensions) wanted.insert(e);
    std::vector<std::string> enabled;
    for (const std::string& e : wanted) {
        if (hasExtension(available, e)) enabled.push_back(e);
        else CRTX_LOG("Extension de périphérique Vulkan indisponible : %s", e.c_str());
    }
    const bool extBda = std::find(enabled.begin(), enabled.end(), "VK_EXT_buffer_device_address") != enabled.end();
    std::vector<const char*> names;
    for (const std::string& e : enabled) names.push_back(e.c_str());

    // Fonctionnalités prises en charge
    VkPhysicalDeviceVulkan12Features supported12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 supported{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    supported.pNext = &supported12;
    vkGetPhysicalDeviceFeatures2(physical_, &supported);
    if (!supported12.timelineSemaphore) throwError("Sémaphores timeline Vulkan non supportés", __FILE__, __LINE__);

    // Fonctionnalités activées
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.timelineSemaphore = VK_TRUE;
    f12.bufferDeviceAddress = (!extBda && supported12.bufferDeviceAddress) ? VK_TRUE : VK_FALSE;
    f12.scalarBlockLayout = supported12.scalarBlockLayout;
    f12.shaderFloat16 = supported12.shaderFloat16;
    f12.hostQueryReset = supported12.hostQueryReset;
    bufferDeviceAddress_ = f12.bufferDeviceAddress == VK_TRUE;
    VkPhysicalDeviceBufferDeviceAddressFeaturesEXT bdaExt{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_BUFFER_DEVICE_ADDRESS_FEATURES_EXT};
    bdaExt.bufferDeviceAddress = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asf.accelerationStructure = VK_TRUE;
    VkPhysicalDeviceRayQueryFeaturesKHR rqf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rqf.rayQuery = VK_TRUE;

    VkPhysicalDeviceFeatures2 features{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    features.features.shaderInt64 = supported.features.shaderInt64;
    features.features.shaderFloat64 = req.float64 ? supported.features.shaderFloat64 : VK_FALSE;
    features.features.shaderStorageImageWriteWithoutFormat = supported.features.shaderStorageImageWriteWithoutFormat;
    features.pNext = &f12;
    void** tail = &f12.pNext;
    if (extBda) {
        *tail = &bdaExt;
        tail = &bdaExt.pNext;
    }
    if (req.rayTracing) {
        *tail = &asf;
        asf.pNext = &rqf;
        tail = &rqf.pNext;
    }

    const float priority = 1.0f;
    VkDeviceQueueCreateInfo qci[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    qci[0].queueFamilyIndex = queueFamily_;
    qci[0].queueCount = 1;
    qci[0].pQueuePriorities = &priority;
    qci[1].queueFamilyIndex = computeFamily_;
    qci[1].queueCount = 1;
    qci[1].pQueuePriorities = &priority;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &features;
    dci.queueCreateInfoCount = asyncCompute ? 2u : 1u;
    dci.pQueueCreateInfos = qci;
    dci.enabledExtensionCount = static_cast<uint32_t>(names.size());
    dci.ppEnabledExtensionNames = names.data();
    VK_CHECK(vkCreateDevice(physical_, &dci, nullptr, &device_));
    vkGetDeviceQueue(device_, queueFamily_, 0, &queue_);
    if (asyncCompute) vkGetDeviceQueue(device_, computeFamily_, 0, &computeQueue_);

    if (req.rayTracing) {
        auto load = [&](auto& fn, const char* name) {
            fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(vkGetDeviceProcAddr(device_, name));
            if (!fn) throwError(std::string("Fonction Vulkan introuvable : ") + name, __FILE__, __LINE__);
        };
        load(rt_.createAS, "vkCreateAccelerationStructureKHR");
        load(rt_.destroyAS, "vkDestroyAccelerationStructureKHR");
        load(rt_.buildSizes, "vkGetAccelerationStructureBuildSizesKHR");
        load(rt_.cmdBuild, "vkCmdBuildAccelerationStructuresKHR");
        load(rt_.asAddress, "vkGetAccelerationStructureDeviceAddressKHR");
        load(rt_.cmdWriteProps, "vkCmdWriteAccelerationStructuresPropertiesKHR");
        load(rt_.cmdCopy, "vkCmdCopyAccelerationStructureKHR");
        VkPhysicalDeviceAccelerationStructurePropertiesKHR asp{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
        VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p2.pNext = &asp;
        vkGetPhysicalDeviceProperties2(physical_, &p2);
        asScratchAlignment_ = std::max(1u, asp.minAccelerationStructureScratchOffsetAlignment);
    }

    VkCommandPoolCreateInfo pci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    pci.queueFamilyIndex = queueFamily_;
    VK_CHECK(vkCreateCommandPool(device_, &pci, nullptr, &commandPool_));

    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateSemaphore(device_, &sci, nullptr, &imageAvailable_));
    createTimelineSemaphore(req.cudaInterop);
}

void VulkanContext::createTimelineSemaphore(bool exportable)
{
    VkExportSemaphoreCreateInfo exp{VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO};
    exp.handleTypes = kSemHandleType;
    VkSemaphoreTypeCreateInfo tci{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
    tci.pNext = exportable ? &exp : nullptr;
    tci.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
    tci.initialValue = 0;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    sci.pNext = &tci;
    VK_CHECK(vkCreateSemaphore(device_, &sci, nullptr, &timeline_));
    if (!exportable) return;

#if defined(CRTX_WITH_CUDA)
    cudaExternalSemaphoreHandleDesc desc{};
#if defined(_WIN32)
    auto getHandle = reinterpret_cast<PFN_vkGetSemaphoreWin32HandleKHR>(
        vkGetDeviceProcAddr(device_, "vkGetSemaphoreWin32HandleKHR"));
    if (!getHandle) throwError("vkGetSemaphoreWin32HandleKHR introuvable", __FILE__, __LINE__);
    VkSemaphoreGetWin32HandleInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_WIN32_HANDLE_INFO_KHR};
    gi.semaphore = timeline_;
    gi.handleType = kSemHandleType;
    HANDLE handle = nullptr;
    VK_CHECK(getHandle(device_, &gi, &handle));
    desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreWin32;
    desc.handle.win32.handle = handle;
    CUDA_CHECK(cudaImportExternalSemaphore(&cudaTimeline_, &desc));
    CloseHandle(handle);
#else
    auto getFd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(vkGetDeviceProcAddr(device_, "vkGetSemaphoreFdKHR"));
    if (!getFd) throwError("vkGetSemaphoreFdKHR introuvable", __FILE__, __LINE__);
    VkSemaphoreGetFdInfoKHR gi{VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR};
    gi.semaphore = timeline_;
    gi.handleType = kSemHandleType;
    int fd = -1;
    VK_CHECK(getFd(device_, &gi, &fd));
    desc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    desc.handle.fd = fd;  // CUDA devient propriétaire du descripteur
    CUDA_CHECK(cudaImportExternalSemaphore(&cudaTimeline_, &desc));
#endif
#else
    throwError("Interopérabilité CUDA demandée dans un build sans CUDA", __FILE__, __LINE__);
#endif
}

void VulkanContext::createSwapchain(uint32_t width, uint32_t height, bool vsync)
{
    if (!surface_) throwError("createSwapchain : contexte sans surface", __FILE__, __LINE__);
    VkSurfaceCapabilitiesKHR caps{};
    VK_CHECK(vkGetPhysicalDeviceSurfaceCapabilitiesKHR(physical_, surface_, &caps));
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_DST_BIT))
        throwError("La swapchain ne supporte pas les copies (TRANSFER_DST)", __FILE__, __LINE__);

    uint32_t fc = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &fc, nullptr);
    std::vector<VkSurfaceFormatKHR> formats(fc);
    vkGetPhysicalDeviceSurfaceFormatsKHR(physical_, surface_, &fc, formats.data());
    VkSurfaceFormatKHR chosen{};
    const VkFormat prefs[] = {VK_FORMAT_B8G8R8A8_UNORM, VK_FORMAT_R8G8B8A8_UNORM, VK_FORMAT_B8G8R8A8_SRGB,
                              VK_FORMAT_R8G8B8A8_SRGB};
    bool ok = false;
    for (VkFormat pf : prefs) {
        for (const auto& f : formats) {
            if (f.format == pf && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) {
                chosen = f;
                ok = true;
                break;
            }
        }
        if (ok) break;
    }
    if (!ok) throwError("Aucun format de swapchain 8 bits BGRA/RGBA disponible", __FILE__, __LINE__);
    swapFormat_ = chosen.format;
    // Les octets sont copiés tels quels : le post-traitement écrit directement en sRGB dans le bon ordre.
    swapBgra_ = chosen.format == VK_FORMAT_B8G8R8A8_UNORM || chosen.format == VK_FORMAT_B8G8R8A8_SRGB;

    uint32_t pc = 0;
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &pc, nullptr);
    std::vector<VkPresentModeKHR> modes(pc);
    vkGetPhysicalDeviceSurfacePresentModesKHR(physical_, surface_, &pc, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync) {
        if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_MAILBOX_KHR) != modes.end())
            mode = VK_PRESENT_MODE_MAILBOX_KHR;
        else if (std::find(modes.begin(), modes.end(), VK_PRESENT_MODE_IMMEDIATE_KHR) != modes.end())
            mode = VK_PRESENT_MODE_IMMEDIATE_KHR;
    }

    VkExtent2D extent = caps.currentExtent;
    if (extent.width == 0xFFFFFFFFu) {
        extent.width = std::clamp(width, caps.minImageExtent.width, caps.maxImageExtent.width);
        extent.height = std::clamp(height, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    uint32_t imageCount = caps.minImageCount + 1;
    if (caps.maxImageCount > 0) imageCount = std::min(imageCount, caps.maxImageCount);

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface_;
    ci.minImageCount = imageCount;
    ci.imageFormat = chosen.format;
    ci.imageColorSpace = chosen.colorSpace;
    ci.imageExtent = extent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_TRANSFER_DST_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    if (!(caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT)) ci.imageUsage &= ~VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    if (!(caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)) {
        for (uint32_t bit = 1; bit != 0; bit <<= 1) {  // premier mode supporté
            if (caps.supportedCompositeAlpha & bit) {
                ci.compositeAlpha = static_cast<VkCompositeAlphaFlagBitsKHR>(bit);
                break;
            }
        }
    }
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain_;
    VkSwapchainKHR newSwap = VK_NULL_HANDLE;
    VK_CHECK(vkCreateSwapchainKHR(device_, &ci, nullptr, &newSwap));
    destroySwapchain();
    swapchain_ = newSwap;
    swapExtent_ = extent;

    uint32_t n = 0;
    vkGetSwapchainImagesKHR(device_, swapchain_, &n, nullptr);
    swapImages_.resize(n);
    vkGetSwapchainImagesKHR(device_, swapchain_, &n, swapImages_.data());
    renderFinished_.resize(n);
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    for (auto& s : renderFinished_) VK_CHECK(vkCreateSemaphore(device_, &sci, nullptr, &s));
    const char* modeName = mode == VK_PRESENT_MODE_MAILBOX_KHR ? "MAILBOX" : mode == VK_PRESENT_MODE_IMMEDIATE_KHR ? "IMMEDIATE" : "FIFO (V-Sync)";
    CRTX_LOG("Swapchain %ux%u, %u images, %s", extent.width, extent.height, n, modeName);
}

void VulkanContext::destroySwapchain()
{
    if (!device_) return;
    for (VkSemaphore s : renderFinished_) vkDestroySemaphore(device_, s, nullptr);
    renderFinished_.clear();
    swapImages_.clear();
    if (swapchain_) vkDestroySwapchainKHR(device_, swapchain_, nullptr);
    swapchain_ = VK_NULL_HANDLE;
}

uint32_t VulkanContext::findMemoryType(uint32_t typeBits, VkMemoryPropertyFlags props) const
{
    VkPhysicalDeviceMemoryProperties mp{};
    vkGetPhysicalDeviceMemoryProperties(physical_, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if ((typeBits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & props) == props) return i;
    throwError("Aucun type de mémoire Vulkan compatible", __FILE__, __LINE__);
}

#if defined(CRTX_WITH_CUDA)
SharedBuffer VulkanContext::createSharedBuffer(VkDeviceSize size)
{
    SharedBuffer b;
    b.size = size;
    VkExternalMemoryBufferCreateInfo ext{VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO};
    ext.handleTypes = kMemHandleType;
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.pNext = &ext;
    bci.size = size;
    bci.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &b.buffer));

    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, b.buffer, &req);
    VkMemoryDedicatedAllocateInfo ded{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_ALLOCATE_INFO};
    ded.buffer = b.buffer;
    VkExportMemoryAllocateInfo exp{VK_STRUCTURE_TYPE_EXPORT_MEMORY_ALLOCATE_INFO};
    exp.pNext = &ded;
    exp.handleTypes = kMemHandleType;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = &exp;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(device_, b.buffer, b.memory, 0));

    cudaExternalMemoryHandleDesc desc{};
    desc.size = req.size;
    desc.flags = cudaExternalMemoryDedicated;
#if defined(_WIN32)
    auto getHandle = reinterpret_cast<PFN_vkGetMemoryWin32HandleKHR>(vkGetDeviceProcAddr(device_, "vkGetMemoryWin32HandleKHR"));
    VkMemoryGetWin32HandleInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_WIN32_HANDLE_INFO_KHR};
    gi.memory = b.memory;
    gi.handleType = kMemHandleType;
    HANDLE handle = nullptr;
    VK_CHECK(getHandle(device_, &gi, &handle));
    desc.type = cudaExternalMemoryHandleTypeOpaqueWin32;
    desc.handle.win32.handle = handle;
    CUDA_CHECK(cudaImportExternalMemory(&b.cudaMem, &desc));
    CloseHandle(handle);
#else
    auto getFd = reinterpret_cast<PFN_vkGetMemoryFdKHR>(vkGetDeviceProcAddr(device_, "vkGetMemoryFdKHR"));
    VkMemoryGetFdInfoKHR gi{VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR};
    gi.memory = b.memory;
    gi.handleType = kMemHandleType;
    int fd = -1;
    VK_CHECK(getFd(device_, &gi, &fd));
    desc.type = cudaExternalMemoryHandleTypeOpaqueFd;
    desc.handle.fd = fd;  // CUDA devient propriétaire du descripteur
    CUDA_CHECK(cudaImportExternalMemory(&b.cudaMem, &desc));
#endif
    cudaExternalMemoryBufferDesc bd{};
    bd.offset = 0;
    bd.size = size;
    CUDA_CHECK(cudaExternalMemoryGetMappedBuffer(&b.cudaPtr, b.cudaMem, &bd));
    return b;
}

void VulkanContext::destroySharedBuffer(SharedBuffer& b)
{
    if (b.cudaPtr) cudaFree(b.cudaPtr);
    if (b.cudaMem) cudaDestroyExternalMemory(b.cudaMem);
    if (b.buffer) vkDestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device_, b.memory, nullptr);
    b = SharedBuffer{};
}
#endif

Buffer VulkanContext::createBuffer(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags props)
{
    Buffer b;
    b.size = std::max<VkDeviceSize>(size, 16);
    VkBufferCreateInfo bci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bci.size = b.size;
    bci.usage = usage;
    bci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(device_, &bci, nullptr, &b.buffer));
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(device_, b.buffer, &req);
    const bool bda = (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) != 0;
    VkMemoryAllocateFlagsInfo flags{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO};
    flags.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.pNext = bda ? &flags : nullptr;
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, props);
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &b.memory));
    VK_CHECK(vkBindBufferMemory(device_, b.buffer, b.memory, 0));
    if (props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) VK_CHECK(vkMapMemory(device_, b.memory, 0, VK_WHOLE_SIZE, 0, &b.mapped));
    if (bda) {
        VkBufferDeviceAddressInfo ai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
        ai.buffer = b.buffer;
        b.address = vkGetBufferDeviceAddress(device_, &ai);
    }
    return b;
}

void VulkanContext::destroyBuffer(Buffer& b)
{
    if (b.mapped) vkUnmapMemory(device_, b.memory);
    if (b.buffer) vkDestroyBuffer(device_, b.buffer, nullptr);
    if (b.memory) vkFreeMemory(device_, b.memory, nullptr);
    b = Buffer{};
}

GpuImage VulkanContext::createImage(uint32_t w, uint32_t h, VkFormat format, VkImageUsageFlags usage)
{
    GpuImage img;
    img.format = format;
    img.width = w;
    img.height = h;
    VkImageCreateInfo ici{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ici.imageType = VK_IMAGE_TYPE_2D;
    ici.format = format;
    ici.extent = {w, h, 1};
    ici.mipLevels = 1;
    ici.arrayLayers = 1;
    ici.samples = VK_SAMPLE_COUNT_1_BIT;
    ici.tiling = VK_IMAGE_TILING_OPTIMAL;
    ici.usage = usage;
    ici.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ici.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK(vkCreateImage(device_, &ici, nullptr, &img.image));
    VkMemoryRequirements req{};
    vkGetImageMemoryRequirements(device_, img.image, &req);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = req.size;
    mai.memoryTypeIndex = findMemoryType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(device_, &mai, nullptr, &img.memory));
    VK_CHECK(vkBindImageMemory(device_, img.image, img.memory, 0));
    VkImageViewCreateInfo vci{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vci.image = img.image;
    vci.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vci.format = format;
    vci.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    VK_CHECK(vkCreateImageView(device_, &vci, nullptr, &img.view));
    return img;
}

void VulkanContext::destroyImage(GpuImage& img)
{
    if (img.view) vkDestroyImageView(device_, img.view, nullptr);
    if (img.image) vkDestroyImage(device_, img.image, nullptr);
    if (img.memory) vkFreeMemory(device_, img.memory, nullptr);
    img = GpuImage{};
}

void VulkanContext::imageBarrier(VkCommandBuffer cmd, VkImage image, VkImageLayout from, VkImageLayout to,
                                 VkPipelineStageFlags srcStage, VkAccessFlags srcAccess, VkPipelineStageFlags dstStage,
                                 VkAccessFlags dstAccess)
{
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    b.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = image;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 0, nullptr, 0, nullptr, 1, &b);
}

void VulkanContext::memoryBarrier(VkCommandBuffer cmd, VkPipelineStageFlags srcStage, VkAccessFlags srcAccess,
                                  VkPipelineStageFlags dstStage, VkAccessFlags dstAccess)
{
    VkMemoryBarrier b{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    b.srcAccessMask = srcAccess;
    b.dstAccessMask = dstAccess;
    vkCmdPipelineBarrier(cmd, srcStage, dstStage, 0, 1, &b, 0, nullptr, 0, nullptr);
}

VkCommandBuffer VulkanContext::beginOneTime()
{
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = commandPool_;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VK_CHECK(vkAllocateCommandBuffers(device_, &ai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void VulkanContext::endOneTime(VkCommandBuffer cmd)
{
    VK_CHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue_));
    vkFreeCommandBuffers(device_, commandPool_, 1, &cmd);
}

void VulkanContext::waitTimeline(uint64_t value)
{
    VkSemaphoreWaitInfo wi{VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO};
    wi.semaphoreCount = 1;
    wi.pSemaphores = &timeline_;
    wi.pValues = &value;
    VK_CHECK(vkWaitSemaphores(device_, &wi, UINT64_MAX));
}

uint64_t VulkanContext::timelineValue() const
{
    uint64_t v = 0;
    vkGetSemaphoreCounterValue(device_, timeline_, &v);
    return v;
}

VkResult VulkanContext::acquireImage(uint32_t& index)
{
    const VkResult r = vkAcquireNextImageKHR(device_, swapchain_, UINT64_MAX, imageAvailable_, VK_NULL_HANDLE, &index);
    if (r != VK_SUCCESS && r != VK_SUBOPTIMAL_KHR && r != VK_ERROR_OUT_OF_DATE_KHR) VK_CHECK(r);
    return r;
}

void VulkanContext::recordCopyToSwapchain(VkCommandBuffer cmd, VkBuffer src, uint32_t index) const
{
    const VkImage img = swapImages_[index];
    imageBarrier(cmd, img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, 0, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy region{};
    region.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    region.imageExtent = {swapExtent_.width, swapExtent_.height, 1};
    vkCmdCopyBufferToImage(cmd, src, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &region);
    imageBarrier(cmd, img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR,
                 VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
}

bool VulkanContext::submitAndPresent(VkCommandBuffer cmd, uint32_t index, uint64_t waitValue, uint64_t signalValue)
{
    VkSemaphore waits[2] = {imageAvailable_, timeline_};
    const VkPipelineStageFlags stages[2] = {VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                                            VK_PIPELINE_STAGE_ALL_COMMANDS_BIT};
    const uint64_t waitVals[2] = {0, waitValue};
    const VkSemaphore signals[2] = {renderFinished_[index], timeline_};
    const uint64_t sigVals[2] = {0, signalValue};
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.waitSemaphoreValueCount = waitValue > 0 ? 2u : 1u;
    ts.pWaitSemaphoreValues = waitVals;
    ts.signalSemaphoreValueCount = 2;
    ts.pSignalSemaphoreValues = sigVals;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ts;
    si.waitSemaphoreCount = waitValue > 0 ? 2u : 1u;
    si.pWaitSemaphores = waits;
    si.pWaitDstStageMask = stages;
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    si.signalSemaphoreCount = 2;
    si.pSignalSemaphores = signals;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));

    VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores = &renderFinished_[index];
    pi.swapchainCount = 1;
    pi.pSwapchains = &swapchain_;
    pi.pImageIndices = &index;
    const VkResult pr = vkQueuePresentKHR(queue_, &pi);
    if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) return false;
    if (pr != VK_SUCCESS) VK_CHECK(pr);
    return true;
}

void VulkanContext::signalTimeline(uint64_t waitValue, uint64_t signalValue)
{
    VkTimelineSemaphoreSubmitInfo ts{VK_STRUCTURE_TYPE_TIMELINE_SEMAPHORE_SUBMIT_INFO};
    ts.waitSemaphoreValueCount = waitValue > 0 ? 1u : 0u;
    ts.pWaitSemaphoreValues = &waitValue;
    ts.signalSemaphoreValueCount = 1;
    ts.pSignalSemaphoreValues = &signalValue;
    const VkPipelineStageFlags stage = VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.pNext = &ts;
    si.waitSemaphoreCount = waitValue > 0 ? 1u : 0u;
    si.pWaitSemaphores = &timeline_;
    si.pWaitDstStageMask = &stage;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores = &timeline_;
    VK_CHECK(vkQueueSubmit(queue_, 1, &si, VK_NULL_HANDLE));
}

}  // namespace crtx
