#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <sys/types.h>
#include <unordered_map>
#include <utility>
#include <vector>

#define VK_ENABLE_BETA_EXTENSIONS
#include <vulkan/vulkan_raii.hpp>

#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#define TINYOBJLOADER_IMPLEMENTATION
#include "tiny_obj_loader.h"

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
const std::string MODEL_PATH = "../assets/models/plant_on_table.obj";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

const std::vector<char const *> validationLayers = {
    "VK_LAYER_KHRONOS_validation"};

// #define NDEBUG
#ifdef NDEBUG
constexpr bool enableValidationLayers = false;
#else
constexpr bool enableValidationLayers = true;
#endif

struct Vertex {
  glm::vec3 pos;
  glm::vec3 color;
  glm::vec2 texCoord;

  static vk::VertexInputBindingDescription getBindingDescription() {
    vk::VertexInputBindingDescription bindingDescription{};
    bindingDescription.binding = 0;
    bindingDescription.stride = sizeof(Vertex);
    bindingDescription.inputRate = vk::VertexInputRate::eVertex;
    return bindingDescription;
  }

  static std::array<vk::VertexInputAttributeDescription, 3>
  getAttributeDescriptions() {
    vk::VertexInputAttributeDescription positionAttributeDescription{};
    positionAttributeDescription.location = 0;
    positionAttributeDescription.binding = 0;
    positionAttributeDescription.format = vk::Format::eR32G32B32Sfloat;
    positionAttributeDescription.offset = offsetof(Vertex, pos);

    vk::VertexInputAttributeDescription colorAttributeDescription{};
    colorAttributeDescription.location = 1;
    colorAttributeDescription.binding = 0;
    colorAttributeDescription.format = vk::Format::eR32G32B32Sfloat;
    colorAttributeDescription.offset = offsetof(Vertex, color);

    vk::VertexInputAttributeDescription texCoordAttributeDescription{};
    texCoordAttributeDescription.location = 2;
    texCoordAttributeDescription.binding = 0;
    texCoordAttributeDescription.format = vk::Format::eR32G32Sfloat;
    texCoordAttributeDescription.offset = offsetof(Vertex, texCoord);

    return {positionAttributeDescription, colorAttributeDescription,
            texCoordAttributeDescription};
  }

  bool operator==(const Vertex &other) const {
    return pos == other.pos && color == other.color &&
           texCoord == other.texCoord;
  }
};

template <> struct std::hash<Vertex> {
  size_t operator()(Vertex const &vertex) const noexcept {
    return ((hash<glm::vec3>()(vertex.pos) ^
             (hash<glm::vec3>()(vertex.color) << 1)) >>
            1) ^
           (hash<glm::vec2>()(vertex.texCoord) << 1);
  }
};

struct UniformBufferObject {
  glm::mat4 model;
  glm::mat4 view;
  glm::mat4 proj;
};

struct PushConstant {
  uint32_t materialIndex;
  uint32_t reflective;
};

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;

  vk::raii::Queue queue = nullptr;
  vk::raii::SwapchainKHR swapChain = nullptr;
  std::vector<vk::Image> swapChainImages;
  vk::SurfaceFormatKHR swapChainSurfaceFormat;
  vk::Extent2D swapChainExtent;
  std::vector<vk::raii::ImageView> swapChainImageViews;

  vk::raii::DescriptorSetLayout descriptorSetLayoutGlobal = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayoutMaterial = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;

  vk::raii::Image depthImage = nullptr;
  vk::raii::DeviceMemory depthImageMemory = nullptr;
  vk::raii::ImageView depthImageView = nullptr;

  vk::raii::Image textureImage = nullptr;
  vk::raii::DeviceMemory textureImageMemory = nullptr;
  std::vector<vk::raii::ImageView> textureImageViews;
  vk::raii::Sampler textureSampler = nullptr;

  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vk::raii::Buffer vertexBuffer = nullptr;
  vk::raii::DeviceMemory vertexBufferMemory = nullptr;
  vk::raii::Buffer indexBuffer = nullptr;
  vk::raii::DeviceMemory indexBufferMemory = nullptr;
  vk::raii::Buffer uvBuffer = nullptr;
  vk::raii::DeviceMemory uvBufferMemory = nullptr;

  std::vector<vk::raii::Buffer> blasBuffers;
  std::vector<vk::raii::DeviceMemory> blasMemories;
  std::vector<vk::raii::AccelerationStructureKHR> blasHandles;

  std::vector<vk::AccelerationStructureInstanceKHR> instances;
  vk::raii::Buffer instanceBuffer = nullptr;
  vk::raii::DeviceMemory instanceMemory = nullptr;

  vk::raii::Buffer tlasBuffer = nullptr;
  vk::raii::DeviceMemory tlasMemory = nullptr;
  vk::raii::Buffer tlasScratchBuffer = nullptr;
  vk::raii::DeviceMemory tlasScratchMemory = nullptr;
  vk::raii::AccelerationStructureKHR tlas = nullptr;

  struct InstanceLUT {
    uint32_t materialID;
    uint32_t indexBufferOffset;
  };

  std::vector<InstanceLUT> instanceLUTs;
  vk::raii::Buffer instanceLUTBuffer = nullptr;
  vk::raii::DeviceMemory instanceLUTBufferMemory = nullptr;

  UniformBufferObject ubo{};

  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  struct SubMesh {
    uint32_t indexOffset;
    uint32_t indexCount;
    int materialID;
    uint32_t firstVertex;
    uint32_t maxVertex;
    bool alphaCut;
    bool reflective;
  };
  std::vector<SubMesh> submeshes;
  std::vector<tinyobj::material_t> materials;

  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> globalDescriptorSets;
  std::vector<vk::raii::DescriptorSet> materialDescriptorSets;

  uint32_t queueIndex = ~0;
  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;

  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
  std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
  std::vector<vk::raii::Fence> inFlightFences;
  uint32_t frameIndex = 0;

  bool framebufferResized = false;

  std::vector<const char *> requiredDeviceExtension = {
      vk::KHRPortabilitySubsetExtensionName,
      vk::KHRSwapchainExtensionName,
      vk::KHRAccelerationStructureExtensionName,
      vk::KHRDeferredHostOperationsExtensionName,
      vk::KHRRayQueryExtensionName,

      /* compatibility extensions if needed
      vk::KHRSynchronization2ExtensionName,
      vk::KHRSpirv14ExtensionName,
      vk::KHRCreateRenderpass2ExtensionName,
      vk::KHRBufferDeviceAddressExtensionName,
      */
  };

  void initWindow() {
    glfwInit();

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    glfwWindowHint(GLFW_RESIZABLE, GLFW_TRUE);

    window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferResizeCallback);
  }

  static void framebufferResizeCallback(GLFWwindow *window, int width,
                                        int height) {
    auto app = static_cast<HelloTriangleApplication *>(
        glfwGetWindowUserPointer(window));
    app->framebufferResized = true;
  }

  void initVulkan() {
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    createLogicalDevice();
    createSwapChain();
    createImageViews();
    createCommandPool();
    loadModel();
    createDescriptorSetLayout();
    createGraphicsPipeline();
    createDepthResources();
    createTextureSampler();

    //        createTextureImage();
    //       createTextureImageView();

    createVertexBuffer();
    createIndexBuffer();

    createUVBuffer();
    createAccelerationStructures();
    // createInstanceLUTBuffer();

    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
  }

  void mainLoop() {
    while (!glfwWindowShouldClose(window)) {
      glfwPollEvents();
      drawFrame();
    }
    device.waitIdle(); // Wait for the device to be idle before exiting to
                       // ensure all resources are cleaned up properly.
  }

  void cleanupSwapChain() {
    swapChainImageViews.clear();
    swapChain = nullptr;
  }

  void cleanup() {
    glfwDestroyWindow(window);

    glfwTerminate();
  }

  void recreateSwapChain() {
    int width = 0, height = 0;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
      glfwGetFramebufferSize(window, &width, &height);
      glfwWaitEvents();
    }

    device.waitIdle();

    cleanupSwapChain();
    createSwapChain();
    createImageViews();
    createDepthResources();
  }

  void createInstance() {
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = vk::ApiVersion14;

    // Get the required layers
    std::vector<char const *> requiredLayers;
    if (enableValidationLayers) {
      requiredLayers.assign(validationLayers.begin(), validationLayers.end());
    }

    // Check if the required layers are supported by the Vulkan implementation.
    auto layerProperties = context.enumerateInstanceLayerProperties();
    auto unsupportedLayerIt = std::ranges::find_if(
        requiredLayers, [&layerProperties](auto const &requiredLayer) {
          return std::ranges::none_of(
              layerProperties, [requiredLayer](auto const &layerProperty) {
                return strcmp(layerProperty.layerName, requiredLayer) == 0;
              });
        });
    if (unsupportedLayerIt != requiredLayers.end()) {
      throw std::runtime_error("Required layer not supported: " +
                               std::string(*unsupportedLayerIt));
    }

    // Get the required extensions.
    auto requiredExtensions = getRequiredInstanceExtensions();
    // MACOS ONLY MoltenVK portability flag
    requiredExtensions.push_back(vk::KHRPortabilityEnumerationExtensionName);

    // Check if the required extensions are supported by the Vulkan
    // implementation.
    auto extensionProperties = context.enumerateInstanceExtensionProperties();
    auto unsupportedPropertyIt = std::ranges::find_if(
        requiredExtensions,
        [&extensionProperties](auto const &requiredExtension) {
          return std::ranges::none_of(
              extensionProperties,
              [requiredExtension](auto const &extensionProperty) {
                return strcmp(extensionProperty.extensionName,
                              requiredExtension) == 0;
              });
        });
    if (unsupportedPropertyIt != requiredExtensions.end()) {
      throw std::runtime_error("Required extension not supported: " +
                               std::string(*unsupportedPropertyIt));
    }

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;
    // Add the portability enumeration flag to enable MoltenVK support on macOS
    createInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    createInfo.enabledLayerCount = static_cast<uint32_t>(requiredLayers.size());
    createInfo.ppEnabledLayerNames = requiredLayers.data();
    createInfo.enabledExtensionCount =
        static_cast<uint32_t>(requiredExtensions.size());
    createInfo.ppEnabledExtensionNames = requiredExtensions.data();

    instance = vk::raii::Instance(context, createInfo);
  }

  void setupDebugMessenger() {
    if (!enableValidationLayers)
      return;

    vk::DebugUtilsMessageSeverityFlagsEXT severityFlags(
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
        vk::DebugUtilsMessageSeverityFlagBitsEXT::eError);
    vk::DebugUtilsMessageTypeFlagsEXT messageTypeFlags(
        vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral |
        vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance |
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation);

    vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMessengerCreateInfoEXT{};
    debugUtilsMessengerCreateInfoEXT.messageSeverity = severityFlags;
    debugUtilsMessengerCreateInfoEXT.messageType = messageTypeFlags;
    debugUtilsMessengerCreateInfoEXT.pfnUserCallback = &debugCallback;
    debugMessenger =
        instance.createDebugUtilsMessengerEXT(debugUtilsMessengerCreateInfoEXT);
  }

  void createSurface() {
    VkSurfaceKHR _surface;
    if (glfwCreateWindowSurface(*instance, window, nullptr, &_surface) != 0) {
      throw std::runtime_error("failed to create window surface!");
    }
    surface = vk::raii::SurfaceKHR(instance, _surface);
  }

  bool isDeviceSuitable(vk::raii::PhysicalDevice const &physicalDevice) {
    // Check if the physicalDevice supports the Vulkan 1.3 API version
    bool supportsVulkan1_3 =
        physicalDevice.getProperties().apiVersion >= vk::ApiVersion13;

    // Check if any of the queue families support graphics operations
    auto queueFamilies = physicalDevice.getQueueFamilyProperties();
    bool supportsGraphics =
        std::ranges::any_of(queueFamilies, [](auto const &qfp) {
          return !!(qfp.queueFlags & vk::QueueFlagBits::eGraphics);
        });

    // Check if all required physicalDevice extensions are available
    auto availableDeviceExtensions =
        physicalDevice.enumerateDeviceExtensionProperties();
    bool supportsAllRequiredExtensions = std::ranges::all_of(
        requiredDeviceExtension,
        [&availableDeviceExtensions](auto const &requiredDeviceExtension) {
          return std::ranges::any_of(
              availableDeviceExtensions,
              [requiredDeviceExtension](auto const &availableDeviceExtension) {
                return strcmp(availableDeviceExtension.extensionName,
                              requiredDeviceExtension) == 0;
              });
        });

    // Check if the physicalDevice supports the required features
    // Check if the physicalDevice supports the required features
    auto features = physicalDevice.template getFeatures2<
        vk::PhysicalDeviceFeatures2, vk::PhysicalDeviceVulkan12Features,
        vk::PhysicalDeviceVulkan13Features,
        vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
        vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
        vk::PhysicalDeviceRayQueryFeaturesKHR>();
    bool supportsRequiredFeatures =
        features.template get<vk::PhysicalDeviceFeatures2>()
            .features.samplerAnisotropy &&
        features.template get<vk::PhysicalDeviceVulkan13Features>()
            .dynamicRendering &&
        features
            .template get<vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT>()
            .extendedDynamicState &&
        features.template get<vk::PhysicalDeviceVulkan12Features>()
            .descriptorBindingSampledImageUpdateAfterBind &&
        features.template get<vk::PhysicalDeviceVulkan12Features>()
            .descriptorBindingPartiallyBound &&
        features.template get<vk::PhysicalDeviceVulkan12Features>()
            .descriptorBindingVariableDescriptorCount &&
        features.template get<vk::PhysicalDeviceVulkan12Features>()
            .runtimeDescriptorArray &&
        features.template get<vk::PhysicalDeviceVulkan12Features>()
            .shaderSampledImageArrayNonUniformIndexing &&
        features.template get<vk::PhysicalDeviceVulkan12Features>()
            .bufferDeviceAddress &&
        features
            .template get<vk::PhysicalDeviceAccelerationStructureFeaturesKHR>()
            .accelerationStructure &&
        features.template get<vk::PhysicalDeviceRayQueryFeaturesKHR>().rayQuery;

    // Return true if the physicalDevice meets all the criteria
    return supportsVulkan1_3 && supportsGraphics &&
           supportsAllRequiredExtensions && supportsRequiredFeatures;
  }

  void pickPhysicalDevice() {

    std::vector<vk::raii::PhysicalDevice> physicalDevices =
        instance.enumeratePhysicalDevices();
    auto const devIter =
        std::ranges::find_if(physicalDevices, [&](auto const &physicalDevice) {
          return isDeviceSuitable(physicalDevice);
        });
    if (devIter == physicalDevices.end()) {
      throw std::runtime_error("failed to find a suitable GPU!");
    }
    physicalDevice = *devIter;
  }

  void createLogicalDevice() {
    std::vector<vk::QueueFamilyProperties> queueFamilyProperties =
        physicalDevice.getQueueFamilyProperties();

    // get the first index into queueFamilyProperties which supports both
    // graphics and present
    for (uint32_t qfpIndex = 0; qfpIndex < queueFamilyProperties.size();
         qfpIndex++) {
      if ((queueFamilyProperties[qfpIndex].queueFlags &
           vk::QueueFlagBits::eGraphics) &&
          physicalDevice.getSurfaceSupportKHR(qfpIndex, *surface)) {
        // found a queue family that supports both graphics and present
        queueIndex = qfpIndex;
        break;
      }
    }
    if (queueIndex == ~0) {
      throw std::runtime_error(
          "Could not find a queue for graphics and present -> terminating");
    }

    // query for Vulkan 1.3 features
    vk::PhysicalDeviceFeatures2 features2{};
    features2.features.samplerAnisotropy =
        true; // Enable anisotropic filtering for better texture quality

    vk::PhysicalDeviceVulkan12Features vulkan12Features{};
    vulkan12Features.descriptorBindingSampledImageUpdateAfterBind = true;
    vulkan12Features.descriptorBindingPartiallyBound = true;
    vulkan12Features.descriptorBindingVariableDescriptorCount = true;
    vulkan12Features.runtimeDescriptorArray = true;
    vulkan12Features.shaderSampledImageArrayNonUniformIndexing = true;
    vulkan12Features.bufferDeviceAddress = true;

    vk::PhysicalDeviceVulkan13Features vulkan13Features{};
    vulkan13Features.dynamicRendering = true;
    vulkan13Features.synchronization2 = true;

    vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
        extendedDynamicStateFeatures{};
    extendedDynamicStateFeatures.extendedDynamicState = true;

    vk::PhysicalDeviceAccelerationStructureFeaturesKHR
        accelerationStructureFeatures{};
    accelerationStructureFeatures.accelerationStructure = true;

    vk::PhysicalDeviceRayQueryFeaturesKHR rayQueryFeatures{};
    rayQueryFeatures.rayQuery = true;

    vk::StructureChain<vk::PhysicalDeviceFeatures2,
                       vk::PhysicalDeviceVulkan12Features,
                       vk::PhysicalDeviceVulkan13Features,
                       vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT,
                       vk::PhysicalDeviceAccelerationStructureFeaturesKHR,
                       vk::PhysicalDeviceRayQueryFeaturesKHR>
        featureChain = {
            features2,        // vk::PhysicalDeviceFeatures2
            vulkan12Features, // vk::PhysicalDeviceVulkan12Features
            vulkan13Features, // vk::PhysicalDeviceVulkan13Features
            extendedDynamicStateFeatures, // vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT
            accelerationStructureFeatures, // vk::PhysicalDeviceAccelerationStructureFeaturesKHR
            rayQueryFeatures // vk::PhysicalDeviceRayQueryFeaturesKHR
        };

    // create a Device
    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
    deviceQueueCreateInfo.queueFamilyIndex = queueIndex;
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

    vk::DeviceCreateInfo deviceCreateInfo{};
    deviceCreateInfo.pNext = &featureChain.get<vk::PhysicalDeviceFeatures2>();
    deviceCreateInfo.queueCreateInfoCount = 1;
    deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
    deviceCreateInfo.enabledExtensionCount =
        static_cast<uint32_t>(requiredDeviceExtension.size());
    deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();

    device = vk::raii::Device(physicalDevice, deviceCreateInfo);
    queue = vk::raii::Queue(device, queueIndex, 0);
  }

  void createSwapChain() {
    vk::SurfaceCapabilitiesKHR surfaceCapabilities =
        physicalDevice.getSurfaceCapabilitiesKHR(*surface);
    swapChainExtent = chooseSwapExtent(surfaceCapabilities);
    uint32_t minImageCount = chooseSwapMinImageCount(surfaceCapabilities);

    std::vector<vk::SurfaceFormatKHR> availableFormats =
        physicalDevice.getSurfaceFormatsKHR(*surface);
    swapChainSurfaceFormat = chooseSwapSurfaceFormat(availableFormats);

    std::vector<vk::PresentModeKHR> availablePresentModes =
        physicalDevice.getSurfacePresentModesKHR(*surface);
    vk::PresentModeKHR presentMode =
        chooseSwapPresentMode(availablePresentModes);

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{};
    swapChainCreateInfo.surface = *surface;
    swapChainCreateInfo.minImageCount = minImageCount;
    swapChainCreateInfo.imageFormat = swapChainSurfaceFormat.format;
    swapChainCreateInfo.imageColorSpace = swapChainSurfaceFormat.colorSpace;
    swapChainCreateInfo.imageExtent = swapChainExtent;
    swapChainCreateInfo.imageArrayLayers = 1;
    swapChainCreateInfo.imageUsage = vk::ImageUsageFlagBits::eColorAttachment;
    swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
    swapChainCreateInfo.preTransform = surfaceCapabilities.currentTransform;
    swapChainCreateInfo.compositeAlpha = vk::CompositeAlphaFlagBitsKHR::eOpaque;
    swapChainCreateInfo.presentMode = presentMode;
    swapChainCreateInfo.clipped = true;

    swapChain = vk::raii::SwapchainKHR(device, swapChainCreateInfo);
    swapChainImages = swapChain.getImages();
  }

  void createImageViews() {
    assert(swapChainImageViews.empty());

    swapChainImageViews.reserve(swapChainImages.size());
    for (auto &image : swapChainImages) {
      swapChainImageViews.emplace_back(
          createImageView(image, swapChainSurfaceFormat.format,
                          vk::ImageAspectFlagBits::eColor));
    }
  }

  void createDescriptorSetLayout() {
    std::array global_bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1,
                                       vk::ShaderStageFlagBits::eVertex |
                                           vk::ShaderStageFlagBits::eFragment,
                                       nullptr),
        vk::DescriptorSetLayoutBinding(
            1, vk::DescriptorType::eAccelerationStructureKHR, 1,
            vk::ShaderStageFlagBits::eFragment, nullptr),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment,
                                       nullptr),
        vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment,
                                       nullptr),
        vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eStorageBuffer, 1,
                                       vk::ShaderStageFlagBits::eFragment,
                                       nullptr)};

    vk::DescriptorSetLayoutCreateInfo globalLayoutInfo{};
    globalLayoutInfo.bindingCount =
        static_cast<uint32_t>(global_bindings.size());
    globalLayoutInfo.pBindings = global_bindings.data();

    descriptorSetLayoutGlobal =
        vk::raii::DescriptorSetLayout(device, globalLayoutInfo);

    // Use descriptor set 1 for bindless material data
    uint32_t textureCount = static_cast<uint32_t>(textureImageViews.size());

    std::array material_bindings = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eSampler, 1,
                                       vk::ShaderStageFlagBits::eFragment,
                                       nullptr),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eSampledImage,
                                       static_cast<uint32_t>(textureCount),
                                       vk::ShaderStageFlagBits::eFragment,
                                       nullptr)};

    std::vector<vk::DescriptorBindingFlags> bindingFlags = {
        vk::DescriptorBindingFlagBits::eUpdateAfterBind,
        vk::DescriptorBindingFlagBits::ePartiallyBound |
            vk::DescriptorBindingFlagBits::eVariableDescriptorCount |
            vk::DescriptorBindingFlagBits::eUpdateAfterBind};

    vk::DescriptorSetLayoutBindingFlagsCreateInfo flagsCreateInfo{
        static_cast<uint32_t>(bindingFlags.size()), bindingFlags.data()};

    vk::DescriptorSetLayoutCreateInfo materialLayoutInfo{
        vk::DescriptorSetLayoutCreateFlagBits::eUpdateAfterBindPool,
        static_cast<uint32_t>(material_bindings.size()),
        material_bindings.data(),
        &flagsCreateInfo,
    };
    descriptorSetLayoutMaterial =
        vk::raii::DescriptorSetLayout(device, materialLayoutInfo);
  }

  void createGraphicsPipeline() {
    vk::raii::ShaderModule shaderModule =
        createShaderModule(readFile("../shaders/simple.spv"));

    vk::PipelineShaderStageCreateInfo vertShaderStageInfo{};
    vertShaderStageInfo.stage = vk::ShaderStageFlagBits::eVertex;
    vertShaderStageInfo.module = shaderModule;
    vertShaderStageInfo.pName = "vertMain";
    vk::PipelineShaderStageCreateInfo fragShaderStageInfo{};
    fragShaderStageInfo.stage = vk::ShaderStageFlagBits::eFragment;
    fragShaderStageInfo.module = shaderModule;
    fragShaderStageInfo.pName = "fragMain";
    vk::PipelineShaderStageCreateInfo shaderStages[] = {vertShaderStageInfo,
                                                        fragShaderStageInfo};

    auto bindingDescription = Vertex::getBindingDescription();
    auto attributeDescriptions = Vertex::getAttributeDescriptions();

    vk::PipelineVertexInputStateCreateInfo vertexInputInfo;
    vertexInputInfo.vertexBindingDescriptionCount = 1;
    vertexInputInfo.pVertexBindingDescriptions = &bindingDescription;
    vertexInputInfo.vertexAttributeDescriptionCount =
        static_cast<uint32_t>(attributeDescriptions.size());
    vertexInputInfo.pVertexAttributeDescriptions = attributeDescriptions.data();

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly{};
    inputAssembly.topology = vk::PrimitiveTopology::eTriangleList;
    vk::PipelineViewportStateCreateInfo viewportState{};
    viewportState.viewportCount = 1;
    viewportState.scissorCount = 1;

    vk::PipelineRasterizationStateCreateInfo rasterizer{};
    rasterizer.depthClampEnable = vk::False;
    rasterizer.rasterizerDiscardEnable = vk::False;
    rasterizer.polygonMode = vk::PolygonMode::eFill;
    rasterizer.cullMode = vk::CullModeFlagBits::eBack;
    rasterizer.frontFace = vk::FrontFace::eCounterClockwise;
    rasterizer.depthBiasEnable = vk::False;
    rasterizer.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisampling{};
    multisampling.rasterizationSamples = vk::SampleCountFlagBits::e1;
    multisampling.sampleShadingEnable = vk::False;

    vk::PipelineDepthStencilStateCreateInfo depthStencil{};
    depthStencil.depthTestEnable = vk::True;
    depthStencil.depthWriteEnable = vk::True;
    depthStencil.depthCompareOp = vk::CompareOp::eLess;
    depthStencil.depthBoundsTestEnable = vk::False;
    depthStencil.stencilTestEnable = vk::False;

    vk::PipelineColorBlendAttachmentState colorBlendAttachment{};
    colorBlendAttachment.blendEnable = vk::False;
    colorBlendAttachment.colorWriteMask =
        vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
        vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA;

    vk::PipelineColorBlendStateCreateInfo colorBlending{};
    colorBlending.logicOpEnable = vk::False;
    colorBlending.logicOp = vk::LogicOp::eCopy;
    colorBlending.attachmentCount = 1;
    colorBlending.pAttachments = &colorBlendAttachment;

    std::vector<vk::DynamicState> dynamicStates = {vk::DynamicState::eViewport,
                                                   vk::DynamicState::eScissor};
    vk::PipelineDynamicStateCreateInfo dynamicState{};
    dynamicState.dynamicStateCount =
        static_cast<uint32_t>(dynamicStates.size());
    dynamicState.pDynamicStates = dynamicStates.data();

    vk::DescriptorSetLayout setLayouts[] = {*descriptorSetLayoutGlobal,
                                            *descriptorSetLayoutMaterial};

    vk::PushConstantRange pushConstantRange{};
    pushConstantRange.stageFlags = vk::ShaderStageFlagBits::eFragment;
    pushConstantRange.offset = 0;
    pushConstantRange.size = sizeof(PushConstant);

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 2;
    pipelineLayoutInfo.pSetLayouts = setLayouts;
    pipelineLayoutInfo.pushConstantRangeCount = 1;
    pipelineLayout = vk::raii::PipelineLayout(device, pipelineLayoutInfo);

    vk::GraphicsPipelineCreateInfo pipelineCreateInfo{};
    pipelineCreateInfo.stageCount = 2;
    pipelineCreateInfo.pStages = shaderStages;
    pipelineCreateInfo.pVertexInputState = &vertexInputInfo;
    pipelineCreateInfo.pInputAssemblyState = &inputAssembly;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pRasterizationState = &rasterizer;
    pipelineCreateInfo.pMultisampleState = &multisampling;
    pipelineCreateInfo.pDepthStencilState = &depthStencil;
    pipelineCreateInfo.pColorBlendState = &colorBlending;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.layout = pipelineLayout;
    pipelineCreateInfo.renderPass = nullptr;

    vk::Format depthFormat = findDepthFormat();
    vk::PipelineRenderingCreateInfo pipelineRenderingCreateInfo{};
    pipelineRenderingCreateInfo.colorAttachmentCount = 1;
    pipelineRenderingCreateInfo.pColorAttachmentFormats =
        &swapChainSurfaceFormat.format;
    pipelineRenderingCreateInfo.depthAttachmentFormat = depthFormat;

    vk::StructureChain<vk::GraphicsPipelineCreateInfo,
                       vk::PipelineRenderingCreateInfo>
        pipelineCreateInfoChain = {
            pipelineCreateInfo,         // vk::GraphicsPipelineCreateInfo
            pipelineRenderingCreateInfo // vk::PipelineRenderingCreateInfo
        };

    graphicsPipeline = vk::raii::Pipeline(
        device, nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = queueIndex;
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }

  void createDepthResources() {
    vk::Format depthFormat = findDepthFormat();
    std::tie(depthImage, depthImageMemory) =
        createImage(swapChainExtent.width, swapChainExtent.height, depthFormat,
                    vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eDepthStencilAttachment,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
    depthImageView = createImageView(depthImage, depthFormat,
                                     vk::ImageAspectFlagBits::eDepth);
  }

  vk::Format findSupportedFormat(const std::vector<vk::Format> candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlagBits features) {
    auto formatIt = std::ranges::find_if(candidates, [&](auto const format) {
      vk::FormatProperties props = physicalDevice.getFormatProperties(format);
      return (((tiling == vk::ImageTiling::eLinear) &&
               ((props.linearTilingFeatures & features) == features)) ||
              ((tiling == vk::ImageTiling::eOptimal) &&
               ((props.optimalTilingFeatures & features) == features)));
    });
    if (formatIt == candidates.end()) {
      throw std::runtime_error("failed to find supported format!");
    }
    return *formatIt;
  }

  vk::Format findDepthFormat() {
    return findSupportedFormat(
        {vk::Format::eD32Sfloat, vk::Format::eD32SfloatS8Uint,
         vk::Format::eD24UnormS8Uint},
        vk::ImageTiling::eOptimal,
        vk::FormatFeatureFlagBits::eDepthStencilAttachment);
  }

  bool hasStencilComponent(vk::Format format) {
    return format == vk::Format::eD32SfloatS8Uint ||
           format == vk::Format::eD24UnormS8Uint;
  }

  std::pair<vk::raii::Buffer, vk::raii::DeviceMemory>
  createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage,
               vk::MemoryPropertyFlags properties) {
    vk::BufferCreateInfo bufferInfo{};
    bufferInfo.size = size;
    bufferInfo.usage = usage;
    bufferInfo.sharingMode = vk::SharingMode::eExclusive;
    vk::raii::Buffer buffer = vk::raii::Buffer(device, bufferInfo);

    vk::MemoryRequirements memRequirements = buffer.getMemoryRequirements();
    vk::MemoryAllocateInfo memoryAllocateInfo{};
    memoryAllocateInfo.allocationSize = memRequirements.size;
    memoryAllocateInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);
    vk::raii::DeviceMemory bufferMemory =
        vk::raii::DeviceMemory(device, memoryAllocateInfo);

    buffer.bindMemory(*bufferMemory, 0);
    return {std::move(buffer), std::move(bufferMemory)};
  }

  std::pair<vk::raii::Image, vk::raii::DeviceMemory>
  createTextureImage(const std::string &path) {
    int texWidth, texHeight, texChannels;
    stbi_uc *pixels = stbi_load(path.c_str(), &texWidth, &texHeight,
                                &texChannels, STBI_rgb_alpha);
    vk::DeviceSize imageSize = texWidth * texHeight * 4;

    if (!pixels) {
      throw std::runtime_error("failed to load texture image!");
    }

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    void *data = stagingBufferMemory.mapMemory(0, imageSize);
    std::memcpy(data, pixels, imageSize);
    stagingBufferMemory.unmapMemory();
    stbi_image_free(pixels);

    std::tie(textureImage, textureImageMemory) = createImage(
        texWidth, texHeight, vk::Format::eR8G8B8A8Srgb,
        vk::ImageTiling::eOptimal,
        vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eSampled,
        vk::MemoryPropertyFlagBits::eDeviceLocal);

    transitionImageLayout(textureImage, vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal);
    copyBufferToImage(stagingBuffer, textureImage,
                      static_cast<uint32_t>(texWidth),
                      static_cast<uint32_t>(texHeight));
    transitionImageLayout(textureImage, vk::ImageLayout::eTransferDstOptimal,
                          vk::ImageLayout::eShaderReadOnlyOptimal);
  }

  vk::raii::ImageView createTextureImageView() {
    return createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb,
                           vk::ImageAspectFlagBits::eColor);
  }

  void createTextureSampler() {
    vk::PhysicalDeviceProperties properties = physicalDevice.getProperties();
    vk::SamplerCreateInfo samplerInfo{};
    samplerInfo.magFilter = vk::Filter::eLinear;
    samplerInfo.minFilter = vk::Filter::eLinear;
    samplerInfo.mipmapMode = vk::SamplerMipmapMode::eLinear;
    samplerInfo.addressModeU = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeV = vk::SamplerAddressMode::eRepeat;
    samplerInfo.addressModeW = vk::SamplerAddressMode::eRepeat;
    samplerInfo.mipLodBias = 0.0f;
    samplerInfo.anisotropyEnable = vk::True;
    samplerInfo.maxAnisotropy = properties.limits.maxSamplerAnisotropy;
    samplerInfo.compareEnable = vk::False;
    samplerInfo.compareOp = vk::CompareOp::eAlways;

    textureSampler = vk::raii::Sampler(device, samplerInfo);
  }

  vk::raii::ImageView createImageView(vk::Image const &image, vk::Format format,
                                      vk::ImageAspectFlags aspectFlags) {
    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.subresourceRange = {aspectFlags, 0, 1, 0, 1};
    return vk::raii::ImageView(device, imageViewCreateInfo);
  }

  std::pair<vk::raii::Image, vk::raii::DeviceMemory>
  createImage(uint32_t width, uint32_t height, vk::Format format,
              vk::ImageTiling tiling, vk::ImageUsageFlags usage,
              vk::MemoryPropertyFlags properties) {
    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format;
    imageInfo.extent = vk::Extent3D{width, height, 1};
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = vk::SampleCountFlagBits::e1;
    imageInfo.tiling = tiling;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;

    vk::raii::Image image = vk::raii::Image(device, imageInfo);

    vk::MemoryRequirements memRequirements = image.getMemoryRequirements();
    vk::MemoryAllocateInfo allocInfo{};
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex =
        findMemoryType(memRequirements.memoryTypeBits, properties);
    vk::raii::DeviceMemory imageMemory =
        vk::raii::DeviceMemory(device, allocInfo);

    image.bindMemory(imageMemory, 0);
    return {std::move(image), std::move(imageMemory)};
  }

  void transitionImageLayout(const vk::raii::Image &image,
                             vk::ImageLayout oldLayout,
                             vk::ImageLayout newLayout) {
    auto commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.layerCount = 1;

    vk::PipelineStageFlags sourceStage;
    vk::PipelineStageFlags destinationStage;

    if (oldLayout == vk::ImageLayout::eUndefined &&
        newLayout == vk::ImageLayout::eTransferDstOptimal) {
      barrier.srcAccessMask = {};
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferWrite;

      sourceStage = vk::PipelineStageFlagBits::eTopOfPipe;
      destinationStage = vk::PipelineStageFlagBits::eTransfer;
    } else if (oldLayout == vk::ImageLayout::eTransferDstOptimal &&
               newLayout == vk::ImageLayout::eShaderReadOnlyOptimal) {
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      sourceStage = vk::PipelineStageFlagBits::eTransfer;
      destinationStage = vk::PipelineStageFlagBits::eFragmentShader;
    } else {
      throw std::invalid_argument("unsupported layout transition!");
    }
    commandBuffer.pipelineBarrier(sourceStage, destinationStage, {}, {}, {},
                                  barrier);
    endSingleTimeCommands(commandBuffer);
  }

  void copyBufferToImage(const vk::raii::Buffer &buffer, vk::raii::Image &image,
                         uint32_t width, uint32_t height) {
    auto commandBuffer = beginSingleTimeCommands();
    vk::BufferImageCopy region{};
    region.bufferOffset = 0;
    region.bufferRowLength = 0;
    region.bufferImageHeight = 0;
    region.imageSubresource.aspectMask = vk::ImageAspectFlagBits::eColor;
    region.imageSubresource.mipLevel = 0;
    region.imageSubresource.baseArrayLayer = 0;
    region.imageSubresource.layerCount = 1;
    region.imageOffset = vk::Offset3D{0, 0, 0};
    region.imageExtent = vk::Extent3D{width, height, 1};

    commandBuffer.copyBufferToImage(
        buffer, image, vk::ImageLayout::eTransferDstOptimal, region);
    endSingleTimeCommands(commandBuffer);
  }

  void loadModel() {
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn, err;

    if (!tinyobj::LoadObj(&attrib, &shapes, &materials, &warn, &err,
                          MODEL_PATH.c_str())) {
      throw std::runtime_error(warn + err);
    }

    std::unordered_map<Vertex, uint32_t> uniqueVertices{};

    for (const auto &shape : shapes) {
      for (const auto &index : shape.mesh.indices) {
        Vertex vertex{};

        vertex.pos = {attrib.vertices[3 * index.vertex_index + 0],
                      attrib.vertices[3 * index.vertex_index + 1],
                      attrib.vertices[3 * index.vertex_index + 2]};

        vertex.texCoord = {attrib.texcoords[2 * index.texcoord_index + 0],
                           1.0f -
                               attrib.texcoords[2 * index.texcoord_index + 1]};

        vertex.color = {1.0f, 1.0f, 1.0f};

#if 1
        auto [it, inserted] = uniqueVertices.insert(
            {vertex, static_cast<uint32_t>(vertices.size())});
        if (inserted) {
          vertices.push_back(vertex);
        }

        indices.push_back(it->second);
#else
        vertices.push_back(vertex);
        indices.push_back(static_cast<uint32_t>(indices.size()));
#endif
      }
    }
  }

  void createVertexBuffer() {
    vk::DeviceSize bufferSize = sizeof(vertices[0]) * vertices.size();

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    void *(dataStaging) = stagingBufferMemory.mapMemory(0, bufferSize);
    std::memcpy(dataStaging, vertices.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(vertexBuffer, vertexBufferMemory) =
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eVertexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(stagingBuffer, vertexBuffer, bufferSize);
  }

  void createIndexBuffer() {
    vk::DeviceSize bufferSize = sizeof(indices[0]) * indices.size();

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    void *data = stagingBufferMemory.mapMemory(0, bufferSize);
    std::memcpy(data, indices.data(), (size_t)bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(indexBuffer, indexBufferMemory) =
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eIndexBuffer |
                         vk::BufferUsageFlagBits::eTransferDst,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    copyBuffer(stagingBuffer, indexBuffer, bufferSize);
  }

  void createUVBuffer() {
    // Extract all texCoords into a separate vector
    std::vector<glm::vec2> uvs;
    uvs.reserve(vertices.size());
    for (auto &v : vertices) {
      uvs.push_back(v.texCoord);
    }

    vk::DeviceSize bufferSize = sizeof(uvs[0]) * uvs.size();

    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(bufferSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    void *dataStaging = stagingBufferMemory.mapMemory(0, bufferSize);
    memcpy(dataStaging, uvs.data(), bufferSize);
    stagingBufferMemory.unmapMemory();

    std::tie(uvBuffer, uvBufferMemory) =
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::eStorageBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);
  }

  void createUniformBuffers() {
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DeviceSize bufferSize = sizeof(UniformBufferObject);
      auto [buffer, bufferMem] =
          createBuffer(bufferSize, vk::BufferUsageFlagBits::eUniformBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible |
                           vk::MemoryPropertyFlagBits::eHostCoherent);
      uniformBuffers.emplace_back(std::move(buffer));
      uniformBuffersMemory.emplace_back(std::move(bufferMem));
      uniformBuffersMapped.emplace_back(
          uniformBuffersMemory.back().mapMemory(0, bufferSize));
    }
  }

  void createAccelerationStructures() {
    vk::BufferDeviceAddressInfo vai{};
    vai.buffer = *vertexBuffer;
    vk::DeviceAddress vertexAddr = device.getBufferAddress(vai);
    vk::BufferDeviceAddressInfo iai{};
    iai.buffer = *indexBuffer;
    vk::DeviceAddress indexAddr = device.getBufferAddress(iai);

    instances.reserve(submeshes.size());
    blasBuffers.reserve(submeshes.size());
    blasMemories.reserve(submeshes.size());
    blasHandles.reserve(submeshes.size());

    vk::TransformMatrixKHR identity{};
    identity.matrix = std::array<std::array<float, 4>, 3>{
        std::array<float, 4>{1.0f, 0.0f, 0.0f, 0.0f},
        std::array<float, 4>{0.0f, 1.0f, 0.0f, 0.0f},
        std::array<float, 4>{0.0f, 0.0f, 1.0f, 0.0f}};
    for (size_t i = 0; i < submeshes.size(); ++i) {
      const auto &submesh = submeshes[i];

      // Prepare the geometry data
      auto trianglesData = vk::AccelerationStructureGeometryTrianglesDataKHR{};
      trianglesData.vertexFormat = vk::Format::eR32G32B32Sfloat;
      trianglesData.vertexData = vertexAddr;
      trianglesData.vertexStride = sizeof(Vertex);
      trianglesData.maxVertex = submesh.maxVertex;
      trianglesData.indexType = vk::IndexType::eUint32;
      trianglesData.indexData =
          indexAddr + submesh.indexOffset * sizeof(uint32_t);

      vk::AccelerationStructureGeometryDataKHR geometryData(trianglesData);

      vk::AccelerationStructureGeometryKHR blasGeometry{};
      blasGeometry.geometryType = vk::GeometryTypeKHR::eTriangles;
      blasGeometry.geometry = geometryData;
      // Update this later
      blasGeometry.flags = vk::GeometryFlagBitsKHR::eOpaque;

      vk::AccelerationStructureBuildGeometryInfoKHR blasBuildGeometryInfo{};
      blasBuildGeometryInfo.type =
          vk::AccelerationStructureTypeKHR::eBottomLevel;
      blasBuildGeometryInfo.mode =
          vk::BuildAccelerationStructureModeKHR::eBuild;
      blasBuildGeometryInfo.geometryCount = 1;
      blasBuildGeometryInfo.pGeometries = &blasGeometry;

      // Query the memory sizes that will be needed for this BLAS
      auto primitiveCount = static_cast<uint32_t>(submesh.indexCount / 3);

      vk::AccelerationStructureBuildSizesInfoKHR blasBuildSizes =
          device.getAccelerationStructureBuildSizesKHR(
              vk::AccelerationStructureBuildTypeKHR::eDevice,
              blasBuildGeometryInfo, {primitiveCount});

      // Create a scratch buffer for the BLAS, this will hold temporary data
      // during the process
      vk::raii::Buffer scratchBuffer = nullptr;
      vk::raii::DeviceMemory scratchMemory = nullptr;
      std::tie(scratchBuffer, scratchMemory) =
          createBuffer(blasBuildSizes.buildScratchSize,
                       vk::BufferUsageFlagBits::eStorageBuffer |
                           vk::BufferUsageFlagBits::eShaderDeviceAddress,
                       vk::MemoryPropertyFlagBits::eDeviceLocal);

      // Save the scratch buffer address in the build info structure
      vk::BufferDeviceAddressInfo scratchAddressInfo{};
      scratchAddressInfo.buffer = *scratchBuffer;
      vk::DeviceAddress scratchAddress =
          device.getBufferAddress(scratchAddressInfo);
      blasBuildGeometryInfo.scratchData.deviceAddress = scratchAddress;

      // Create a buffer for the BLAS itself now that we know the required
      // size
      vk::raii::Buffer blasBuffer = nullptr;
      vk::raii::DeviceMemory blasBufferMemory = nullptr;
      blasBuffers.emplace_back(std::move(blasBuffer));
      blasMemories.emplace_back(std::move(blasBufferMemory));
      std::tie(blasBuffers[i], blasMemories[i]) = createBuffer(
          blasBuildSizes.accelerationStructureSize,
          vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
              vk::BufferUsageFlagBits::eShaderDeviceAddress,
          vk::MemoryPropertyFlagBits::eDeviceLocal);

      // Create and store the BLAS handle
      vk::AccelerationStructureCreateInfoKHR blasCreateInfo{};
      blasCreateInfo.buffer = blasBuffers[i];
      blasCreateInfo.offset = 0;
      blasCreateInfo.size = blasBuildSizes.accelerationStructureSize;
      blasCreateInfo.type = vk::AccelerationStructureTypeKHR::eBottomLevel;
      blasHandles.emplace_back(
          device.createAccelerationStructureKHR(blasCreateInfo));

      blasBuildGeometryInfo.dstAccelerationStructure = blasHandles[i];

      // Prepare the build range for the BLAS
      vk::AccelerationStructureBuildRangeInfoKHR blasRangeInfo{};
      blasRangeInfo.primitiveCount = primitiveCount;
      blasRangeInfo.primitiveOffset = 0;
      blasRangeInfo.firstVertex = submesh.firstVertex;
      blasRangeInfo.transformOffset = 0;

      // Build the BLAS
      auto cmd = beginSingleTimeCommands();
      cmd.buildAccelerationStructuresKHR({blasBuildGeometryInfo},
                                         {&blasRangeInfo});
      endSingleTimeCommands(cmd);

      // Create a BLAS instance for the TLAS
      vk::AccelerationStructureDeviceAddressInfoKHR addrInfo{};
      addrInfo.accelerationStructure = *blasHandles[i];
      vk::DeviceAddress blasDeviceAddr =
          device.getAccelerationStructureAddressKHR(addrInfo);

      vk::AccelerationStructureInstanceKHR instance{};
      instance.transform = identity;
      instance.mask = 0xFF;
      instance.accelerationStructureReference = blasDeviceAddr;
      instances.push_back(instance);
    }

    // Prepare the instance data buffer
    vk::DeviceSize instBufferSize = sizeof(instances[0]) * instances.size();
    auto [instanceBuffer, instanceMemory] =
        createBuffer(instBufferSize,
                     vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::eTransferDst |
                         vk::BufferUsageFlagBits::
                             eAccelerationStructureBuildInputReadOnlyKHR,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    void *ptr = instanceMemory.mapMemory(0, instBufferSize);
    std::memcpy(ptr, instances.data(), instBufferSize);
    instanceMemory.unmapMemory();

    vk::BufferDeviceAddressInfo instanceAddrInfo{};
    instanceAddrInfo.buffer = instanceBuffer;
    vk::DeviceAddress instanceAddr =
        device.getBufferAddressKHR(instanceAddrInfo);

    // Prepare the geometry (instance) data
    vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};
    instancesData.arrayOfPointers = vk::False;
    instancesData.data = instanceAddr;

    vk::AccelerationStructureGeometryDataKHR geometryData(instancesData);

    vk::AccelerationStructureGeometryKHR tlasGeometry{};
    tlasGeometry.geometryType = vk::GeometryTypeKHR::eInstances;
    tlasGeometry.geometry = geometryData;

    vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildGeometryInfo{};
    tlasBuildGeometryInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    tlasBuildGeometryInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    tlasBuildGeometryInfo.geometryCount = 1;
    tlasBuildGeometryInfo.pGeometries = &tlasGeometry;

    // Query the memory sizes that will be needed for this TLAS
    auto primitiveCount = static_cast<uint32_t>(instances.size());

    vk::AccelerationStructureBuildSizesInfoKHR tlasBuildSizes =
        device.getAccelerationStructureBuildSizesKHR(
            vk::AccelerationStructureBuildTypeKHR::eDevice,
            tlasBuildGeometryInfo, {primitiveCount});

    // Create a scratch buffer for the TLAS, this will hold temporary data
    // during the build process
    std::tie(tlasScratchBuffer, tlasScratchMemory) =
        createBuffer(tlasBuildSizes.buildScratchSize,
                     vk::BufferUsageFlagBits::eStorageBuffer |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress,
                     vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Save the scratch buffer address in the build info structure
    vk::BufferDeviceAddressInfo scratchAddressInfo{};
    scratchAddressInfo.buffer = *tlasScratchBuffer;
    vk::DeviceAddress scratchAddr =
        device.getBufferAddressKHR(scratchAddressInfo);
    tlasBuildGeometryInfo.scratchData.deviceAddress = scratchAddr;

    // Create a buffer for the TLAS itself now that we now the required size
    std::tie(tlasBuffer, tlasMemory) =
        createBuffer(tlasBuildSizes.accelerationStructureSize,
                     vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR |
                         vk::BufferUsageFlagBits::eShaderDeviceAddress |
                         vk::BufferUsageFlagBits::
                             eAccelerationStructureBuildInputReadOnlyKHR,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);

    // Create and store the TLAS handle
    vk::AccelerationStructureCreateInfoKHR tlasCreateInfo{};
    tlasCreateInfo.buffer = tlasBuffer;
    tlasCreateInfo.offset = 0;
    tlasCreateInfo.size = tlasBuildSizes.accelerationStructureSize,
    tlasCreateInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    tlas = device.createAccelerationStructureKHR(tlasCreateInfo);

    // Save the TLAS handle in the build info structure
    tlasBuildGeometryInfo.dstAccelerationStructure = tlas;

    // Prepare the build range for the TLAS
    vk::AccelerationStructureBuildRangeInfoKHR tlasRangeInfo{};
    tlasRangeInfo.primitiveCount = primitiveCount;
    tlasRangeInfo.primitiveOffset = 0;
    tlasRangeInfo.firstVertex = 0;
    tlasRangeInfo.transformOffset = 0;

    // Build the TLAS
    auto cmd = beginSingleTimeCommands();
    cmd.buildAccelerationStructuresKHR({tlasBuildGeometryInfo},
                                       {&tlasRangeInfo});
    endSingleTimeCommands(cmd);
  }

  void createDescriptorPool() {
    std::array poolSize{
        vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer,
                               MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eAccelerationStructureKHR,
                               MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer,
                               MAX_FRAMES_IN_FLIGHT *
                                   3), // indices, UVs, instance LUT
        vk::DescriptorPoolSize(vk::DescriptorType::eSampler,
                               MAX_FRAMES_IN_FLIGHT),
        vk::DescriptorPoolSize(vk::DescriptorType::eSampledImage,
                               (uint32_t)materials.size())};

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet |
                     vk::DescriptorPoolCreateFlagBits::eUpdateAfterBind,
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
    poolInfo.pPoolSizes = poolSize.data();

    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
  }

  void createDescriptorSets() {
    // Global descriptor sets (per frame)
    std::vector<vk::DescriptorSetLayout> globalLayouts(
        MAX_FRAMES_IN_FLIGHT, descriptorSetLayoutGlobal);
    vk::DescriptorSetAllocateInfo allocInfoGlobal{};
    allocInfoGlobal.descriptorPool = descriptorPool;
    allocInfoGlobal.descriptorSetCount =
        static_cast<uint32_t>(globalLayouts.size());
    allocInfoGlobal.pSetLayouts = globalLayouts.data();

    globalDescriptorSets.clear();
    globalDescriptorSets = device.allocateDescriptorSets(allocInfoGlobal);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      // UBO descriptor
      vk::DescriptorBufferInfo bufferInfo{};
      bufferInfo.buffer = uniformBuffers[i];
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject);

      vk::WriteDescriptorSet bufferWrite{};
      bufferWrite.dstSet = globalDescriptorSets[i];
      bufferWrite.dstBinding = 0;
      bufferWrite.dstArrayElement = 0;
      bufferWrite.descriptorCount = 1;
      bufferWrite.descriptorType = vk::DescriptorType::eUniformBuffer;
      bufferWrite.pBufferInfo = &bufferInfo;

      // Acceleration structure descriptor
      vk::WriteDescriptorSetAccelerationStructureKHR asInfo{1, {&*tlas}};
      vk::WriteDescriptorSet asWrite{};
      asWrite.pNext = &asInfo;
      asWrite.dstSet = globalDescriptorSets[i];
      asWrite.dstBinding = 1;
      asWrite.dstArrayElement = 0;
      asWrite.descriptorCount = 1;
      asWrite.descriptorType = vk::DescriptorType::eAccelerationStructureKHR;

      // indices SSBO
      vk::DescriptorBufferInfo indexBufferInfo{};
      indexBufferInfo.buffer = indexBuffer;
      indexBufferInfo.offset = 0;
      indexBufferInfo.range = sizeof(uint32_t) * indices.size();

      vk::WriteDescriptorSet indexBufferWrite{};
      indexBufferWrite.dstSet = globalDescriptorSets[i];
      indexBufferWrite.dstBinding = 2;
      indexBufferWrite.dstArrayElement = 0;
      indexBufferWrite.descriptorCount = 1;
      indexBufferWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
      indexBufferWrite.pBufferInfo = &indexBufferInfo;

      // UVs SSBOs
      vk::DescriptorBufferInfo uvBufferInfo{};
      uvBufferInfo.buffer = indexBuffer;
      uvBufferInfo.offset = 0;
      uvBufferInfo.range = sizeof(uint32_t) * indices.size();

      vk::WriteDescriptorSet uvBufferWrite{};
      uvBufferWrite.dstSet = globalDescriptorSets[i];
      uvBufferWrite.dstBinding = 3;
      uvBufferWrite.dstArrayElement = 0;
      uvBufferWrite.descriptorCount = 1;
      uvBufferWrite.descriptorType = vk::DescriptorType::eStorageBuffer;
      uvBufferWrite.pBufferInfo = &uvBufferInfo;

      std::array<vk::WriteDescriptorSet, 4> descriptorWrites{
          bufferWrite, asWrite, indexBufferWrite, uvBufferWrite};
      device.updateDescriptorSets(descriptorWrites, {});
    }

    // Material descriptor sets (per material)
    std::vector<uint32_t> variableCounts = {
        static_cast<uint32_t>(textureImageViews.size())};

    vk::DescriptorSetVariableDescriptorCountAllocateInfo variableCountInfo{};
    variableCountInfo.descriptorSetCount = 1;
    variableCountInfo.pDescriptorCounts = variableCounts.data();

    std::vector<vk::DescriptorSetLayout> layouts{*descriptorSetLayoutMaterial};

    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.pNext = &variableCountInfo;
    allocInfo.descriptorPool = descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    materialDescriptorSets = device.allocateDescriptorSets(allocInfo);

    // Sampler
    vk::DescriptorImageInfo samplerInfo{textureSampler};
    vk::WriteDescriptorSet samplerWrite{};
    samplerWrite.dstSet = materialDescriptorSets[0];
    samplerWrite.dstBinding = 0;
    samplerWrite.dstArrayElement = 0;
    samplerWrite.descriptorCount = 1;
    samplerWrite.descriptorType = vk::DescriptorType::eSampler;
    samplerWrite.pImageInfo = &samplerInfo;

    device.updateDescriptorSets({samplerWrite}, {});

    // Textures
    std::vector<vk::DescriptorImageInfo> imageInfos;
    imageInfos.reserve(textureImageViews.size());
    for (auto &iv : textureImageViews) {
      vk::DescriptorImageInfo imageInfo{};
      imageInfo.imageView = iv;
      imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      imageInfos.push_back(imageInfo);
    }

    vk::WriteDescriptorSet materialWrite{};

    materialWrite.dstSet = materialDescriptorSets[0];
    materialWrite.dstBinding = 1;
    materialWrite.dstArrayElement = 0;
    materialWrite.descriptorCount = static_cast<uint32_t>(imageInfos.size());
    materialWrite.descriptorType = vk::DescriptorType::eSampledImage;
    materialWrite.pImageInfo = imageInfos.data();

    device.updateDescriptorSets({materialWrite}, {});
  }

  vk::raii::CommandBuffer beginSingleTimeCommands() {
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = 1;
    vk::raii::CommandBuffer commandBuffer =
        std::move(vk::raii::CommandBuffers(device, allocInfo).front());

    vk::CommandBufferBeginInfo beginInfo{};
    beginInfo.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit;
    commandBuffer.begin(beginInfo);

    return commandBuffer;
  }

  void
  endSingleTimeCommands(const vk::raii::CommandBuffer &commandBuffer) const {
    commandBuffer.end();

    vk::SubmitInfo submitInfo{};
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandBuffer;
    queue.submit(submitInfo, nullptr);
    queue.waitIdle();
  }

  void copyBuffer(vk::raii::Buffer &srcBuffer, vk::raii::Buffer &dstBuffer,
                  vk::DeviceSize size) {
    vk::raii::CommandBuffer commandCopyBuffer = beginSingleTimeCommands();
    commandCopyBuffer.copyBuffer(*srcBuffer, *dstBuffer,
                                 vk::BufferCopy(0, 0, size));
    endSingleTimeCommands(commandCopyBuffer);
  }

  uint32_t findMemoryType(uint32_t typeFilter,
                          vk::MemoryPropertyFlags properties) {
    vk::PhysicalDeviceMemoryProperties memProperties =
        physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
      if ((typeFilter & (1 << i)) &&
          (memProperties.memoryTypes[i].propertyFlags & properties) ==
              properties) {
        return i;
      }
    }

    throw std::runtime_error("Failed to find suitable memory type.");
  }

  void createCommandBuffers() {
    commandBuffers.clear();
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    auto &commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin({});

    // Before starting rendering, transition the swapchain image to
    // vk::ImageLayout::eColorAttachmentOptimal
    transition_image_layout(swapChainImages[imageIndex],
                            vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal, {},
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::ImageAspectFlagBits::eColor);

    // Transition depth image to depth attachment optimal layout
    transition_image_layout(depthImage, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eDepthStencilAttachmentOptimal,
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits2::eLateFragmentTests,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits2::eLateFragmentTests,
                            vk::ImageAspectFlagBits::eDepth);

    vk::ClearValue clearColor = vk::ClearColorValue(0.0f, 0.0f, 0.0f, 1.0f);
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);

    vk::RenderingAttachmentInfo colorAttachmentInfo = {};
    colorAttachmentInfo.imageView = swapChainImageViews[imageIndex];
    colorAttachmentInfo.imageLayout = vk::ImageLayout::eColorAttachmentOptimal;
    colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentInfo.clearValue = clearColor;

    vk::RenderingAttachmentInfo depthAttachmentInfo = {};
    depthAttachmentInfo.imageView = depthImageView;
    depthAttachmentInfo.imageLayout = vk::ImageLayout::eDepthAttachmentOptimal;
    depthAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachmentInfo.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachmentInfo.clearValue = clearDepth;

    vk::RenderingInfo renderingInfo = {};
    renderingInfo.renderArea.offset = {.x = 0, .y = 0};
    renderingInfo.renderArea.extent = swapChainExtent;
    renderingInfo.layerCount = 1;
    renderingInfo.colorAttachmentCount = 1;
    renderingInfo.pColorAttachments = &colorAttachmentInfo;
    renderingInfo.pDepthAttachment = &depthAttachmentInfo;

    commandBuffer.beginRendering(renderingInfo);
    commandBuffer.bindPipeline(vk::PipelineBindPoint::eGraphics,
                               *graphicsPipeline);
    commandBuffer.setViewport(
        0,
        vk::Viewport(0.0f, 0.0f, static_cast<float>(swapChainExtent.width),
                     static_cast<float>(swapChainExtent.height), 0.0f, 1.0f));
    commandBuffer.setScissor(0,
                             vk::Rect2D(vk::Offset2D(0, 0), swapChainExtent));
    commandBuffer.bindVertexBuffers(0, *vertexBuffer, {0});
    commandBuffer.bindIndexBuffer(
        *indexBuffer, 0,
        vk::IndexTypeValue<decltype(indices)::value_type>::value);
    commandBuffer.bindDescriptorSets(
        vk::PipelineBindPoint::eGraphics, pipelineLayout, 0,
        *globalDescriptorSets[frameIndex], nullptr);
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                     pipelineLayout, 1,
                                     *materialDescriptorSets[0], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0,
                              0);
    commandBuffer.endRendering();

    // After rendering, transition the swapchain image to
    // vk::ImageLayout::ePresentSrcKHR
    transition_image_layout(swapChainImages[imageIndex],
                            vk::ImageLayout::eColorAttachmentOptimal,
                            vk::ImageLayout::ePresentSrcKHR,
                            vk::AccessFlagBits2::eColorAttachmentWrite, {},
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eBottomOfPipe,
                            vk::ImageAspectFlagBits::eColor);
    commandBuffer.end();
  }

  void transition_image_layout(vk::Image image, vk::ImageLayout oldLayout,
                               vk::ImageLayout newLayout,
                               vk::AccessFlags2 srcAccessMask,
                               vk::AccessFlags2 dstAccessMask,
                               vk::PipelineStageFlags2 srcStageMask,
                               vk::PipelineStageFlags2 dstStageMask,
                               vk::ImageAspectFlagBits imageAspectFlags) {
    vk::ImageMemoryBarrier2 barrier{};
    barrier.srcStageMask = srcStageMask;
    barrier.srcAccessMask = srcAccessMask;
    barrier.dstStageMask = dstStageMask;
    barrier.dstAccessMask = dstAccessMask;
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = imageAspectFlags;
    barrier.subresourceRange.baseMipLevel = 0;
    barrier.subresourceRange.levelCount = 1;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;

    vk::DependencyInfo dependencyInfo{};
    dependencyInfo.dependencyFlags = {};
    dependencyInfo.imageMemoryBarrierCount = 1;
    dependencyInfo.pImageMemoryBarriers = &barrier;

    commandBuffers[frameIndex].pipelineBarrier2(dependencyInfo);
  }

  void createSyncObjects() {

    assert(presentCompleteSemaphores.empty() &&
           renderFinishedSemaphores.empty() && inFlightFences.empty());
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      renderFinishedSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
    }
    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      presentCompleteSemaphores.emplace_back(device, vk::SemaphoreCreateInfo());
      vk::FenceCreateInfo fenceCreateInfo{};
      fenceCreateInfo.flags = vk::FenceCreateFlagBits::eSignaled;
      inFlightFences.emplace_back(device, fenceCreateInfo);
    }
  }

  void updateUniformBuffer(uint32_t currentImage) {
    static auto startTime = std::chrono::high_resolution_clock::now();
    auto currentTime = std::chrono::high_resolution_clock::now();
    float time = std::chrono::duration<float, std::chrono::seconds::period>(
                     currentTime - startTime)
                     .count();

    UniformBufferObject ubo{};
    ubo.model = glm::rotate(glm::mat4(1.0f), time * glm::radians(90.0f),
                            glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.view =
        glm::lookAt(glm::vec3(2.0f, 2.0f, 2.0f), glm::vec3(0.0f, 0.0f, 0.0f),
                    glm::vec3(0.0f, 0.0f, 1.0f));
    ubo.proj = glm::perspective(glm::radians(45.0f),
                                static_cast<float>(swapChainExtent.width) /
                                    static_cast<float>(swapChainExtent.height),
                                0.1f, 10.0f);
    ubo.proj[1][1] *= -1; // Invert Y coordinate of the clip space

    std::memcpy(uniformBuffersMapped[currentImage], &ubo, sizeof(ubo));
  }

  void updateTopLevelAS(const glm::mat4 &model) {}

  void drawFrame() {
    // Note: inFlightFences, presentCompleteSemaphores, and commandBuffers
    // are indexed by frameIndex,
    //       while renderFinishedSemaphores is indexed by imageIndex
    auto fenceResult =
        device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to wait for draw fence!");
    }

    auto [result, imageIndex] = swapChain.acquireNextImage(
        UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

    // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined,
    // eErrorOutOfDateKHR can be checked as a result here and does not need
    // to be caught by an exception.
    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    }

    // On other success codes than eSuccess and eSuboptimalKHR we just throw
    // an exception. On any error code, aquireNextImage already threw an
    // exception.
    if (result != vk::Result::eSuccess &&
        result != vk::Result::eSuboptimalKHR) {
      assert(result == vk::Result::eTimeout || result == vk::Result::eNotReady);
      throw std::runtime_error("failed to acquire swap chain image!");
    }
    updateUniformBuffer(frameIndex);

    // Only reset the fence if we are submitting work
    device.resetFences(*inFlightFences[frameIndex]);

    commandBuffers[frameIndex].reset();
    recordCommandBuffer(imageIndex);

    queue.waitIdle(); // NOTE: for simplicity, wait for the queue to be idle
                      // before starting the frame
    // In the next chapter you see how to use multiple frames in flight and
    // fences to sync

    vk::PipelineStageFlags waitDestionationStageMask =
        vk::PipelineStageFlagBits::eColorAttachmentOutput;
    vk::SubmitInfo submitInfo{};
    submitInfo.waitSemaphoreCount = 1;
    submitInfo.pWaitSemaphores = &*presentCompleteSemaphores[frameIndex];
    submitInfo.pWaitDstStageMask = &waitDestionationStageMask;
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &*commandBuffers[frameIndex];
    submitInfo.signalSemaphoreCount = 1;
    submitInfo.pSignalSemaphores = &*renderFinishedSemaphores[imageIndex];

    queue.submit(submitInfo, *inFlightFences[frameIndex]);

    vk::PresentInfoKHR presentInfoKHR{};
    presentInfoKHR.waitSemaphoreCount = 1;
    presentInfoKHR.pWaitSemaphores = &*renderFinishedSemaphores[imageIndex];
    presentInfoKHR.swapchainCount = 1;
    presentInfoKHR.pSwapchains = &*swapChain;
    presentInfoKHR.pImageIndices = &imageIndex;

    result = queue.presentKHR(presentInfoKHR);
    // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined,
    // eErrorOutOfDateKHR can be checked as a result here and does not need
    // to be caught by an exception.
    if ((result == vk::Result::eSuboptimalKHR) ||
        (result == vk::Result::eErrorOutOfDateKHR) || framebufferResized) {
      framebufferResized = false;
      recreateSwapChain();
      std::cout << "Swap chain recreated due to "
                << (result == vk::Result::eSuboptimalKHR
                        ? "suboptimal swap chain"
                        : "out of date swap chain")
                << std::endl;
    } else {
      // There are no other success codes than eSuccess; on any error code,
      // presentKHR already threw an exception.
      assert(result == vk::Result::eSuccess);
    }

    frameIndex = (frameIndex + 1) % MAX_FRAMES_IN_FLIGHT;
  }

  [[nodiscard]] vk::raii::ShaderModule
  createShaderModule(const std::vector<char> &code) const {
    vk::ShaderModuleCreateInfo createInfo{};
    createInfo.codeSize = code.size() * sizeof(char);
    createInfo.pCode = reinterpret_cast<const uint32_t *>(code.data());

    vk::raii::ShaderModule shaderModule{device, createInfo};

    return shaderModule;
  }

  static uint32_t chooseSwapMinImageCount(
      vk::SurfaceCapabilitiesKHR const &surfaceCapabilities) {
    auto minImageCount = std::max(3u, surfaceCapabilities.minImageCount);
    if ((0 < surfaceCapabilities.maxImageCount) &&
        (surfaceCapabilities.maxImageCount < minImageCount)) {
      minImageCount = surfaceCapabilities.maxImageCount;
    }
    return minImageCount;
  }

  static vk::SurfaceFormatKHR chooseSwapSurfaceFormat(
      const std::vector<vk::SurfaceFormatKHR> &availableFormats) {
    assert(!availableFormats.empty());
    const auto formatIt =
        std::ranges::find_if(availableFormats, [](const auto &format) {
          return format.format == vk::Format::eB8G8R8A8Srgb &&
                 format.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear;
        });
    return formatIt != availableFormats.end() ? *formatIt : availableFormats[0];
  }

  static vk::PresentModeKHR chooseSwapPresentMode(
      std::vector<vk::PresentModeKHR> const &availablePresentModes) {
    assert(std::ranges::any_of(availablePresentModes, [](auto presentMode) {
      return presentMode == vk::PresentModeKHR::eFifo;
    }));
    return std::ranges::any_of(availablePresentModes,
                               [](const vk::PresentModeKHR value) {
                                 return vk::PresentModeKHR::eMailbox == value;
                               })
               ? vk::PresentModeKHR::eMailbox
               : vk::PresentModeKHR::eFifo;
  }

  vk::Extent2D
  chooseSwapExtent(vk::SurfaceCapabilitiesKHR const &capabilities) {
    if (capabilities.currentExtent.width !=
        std::numeric_limits<uint32_t>::max()) {
      return capabilities.currentExtent;
    }
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);

    return {std::clamp<uint32_t>(width, capabilities.minImageExtent.width,
                                 capabilities.maxImageExtent.width),
            std::clamp<uint32_t>(height, capabilities.minImageExtent.height,
                                 capabilities.maxImageExtent.height)};
  }

  std::vector<const char *> getRequiredInstanceExtensions() {
    uint32_t glfwExtensionCount = 0;
    auto glfwExtensions =
        glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

    std::vector extensions(glfwExtensions, glfwExtensions + glfwExtensionCount);
    if (enableValidationLayers) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    }

    return extensions;
  }

  static VKAPI_ATTR vk::Bool32 VKAPI_CALL debugCallback(
      vk::DebugUtilsMessageSeverityFlagBitsEXT severity,
      vk::DebugUtilsMessageTypeFlagsEXT type,
      const vk::DebugUtilsMessengerCallbackDataEXT *pCallbackData, void *) {
    if (severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eError ||
        severity == vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning) {
      std::cerr << "validation layer: type " << to_string(type)
                << " msg: " << pCallbackData->pMessage << std::endl;
    }

    return vk::False;
  }

  static std::vector<char> readFile(const std::string &filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
      throw std::runtime_error("failed to open file!");
    }
    std::vector<char> buffer(file.tellg());
    file.seekg(0, std::ios::beg);
    file.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
    file.close();
    return buffer;
  }
};

int main() {
  try {
    HelloTriangleApplication app;
    app.run();
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
