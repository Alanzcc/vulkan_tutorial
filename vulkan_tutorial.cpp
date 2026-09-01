// STL
#include <algorithm>
#include <array>
#include <assert.h>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// Graphics API
#define VK_ENABLE_BETA_EXTENSIONS
#include "profiles/vulkan_profiles.hpp"
#include <vulkan/vulkan_raii.hpp>

// Window manager
#define GLFW_INCLUDE_VULKAN // REQUIRED only for GLFW CreateWindowSurface.
#include <GLFW/glfw3.h>

// GLM
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#define GLM_ENABLE_EXPERIMENTAL
#include <glm/gtx/hash.hpp>

#include <fastgltf/core.hpp>
#include <fastgltf/glm_element_traits.hpp> // lets iterateAccessor decode straight into glm::vec3/vec2
#include <fastgltf/tools.hpp>
// Texture loader
#include <ktx.h>

constexpr uint32_t WIDTH = 800;
constexpr uint32_t HEIGHT = 600;
const std::filesystem::path MODEL_PATH = "../assets/models/viking_room.glb";
const std::string TEXTURE_PATH = "../assets/textures/viking_room.ktx2";
constexpr int MAX_FRAMES_IN_FLIGHT = 2;

// Application info structure to store profile support flags
struct AppInfo {
  bool profileSupported = false;
  VpProfileProperties profile;
};

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

class HelloTriangleApplication {
public:
  void run() {
    initWindow();
    initVulkan();
    mainLoop();
    cleanup();
  }

private:
  // Setup
  GLFWwindow *window = nullptr;
  vk::raii::Context context;
  vk::raii::Instance instance = nullptr;
  vk::raii::DebugUtilsMessengerEXT debugMessenger = nullptr;
  vk::raii::SurfaceKHR surface = nullptr;
  vk::raii::PhysicalDevice physicalDevice = nullptr;
  vk::raii::Device device = nullptr;
  uint32_t queueIndex = ~0;
  vk::raii::Queue queue = nullptr;
  vk::raii::SwapchainKHR swapChain = nullptr;
  std::vector<vk::Image> swapChainImages;
  vk::SurfaceFormatKHR swapChainSurfaceFormat;
  vk::Extent2D swapChainExtent;
  std::vector<vk::raii::ImageView> swapChainImageViews;

  // Rendering
  vk::raii::RenderPass renderPass = nullptr;
  vk::raii::DescriptorSetLayout descriptorSetLayout = nullptr;
  vk::raii::PipelineLayout pipelineLayout = nullptr;
  vk::raii::Pipeline graphicsPipeline = nullptr;
  std::vector<vk::raii::Framebuffer> swapChainFramebuffers;

  // MSAA
  vk::SampleCountFlagBits msaaSamples = vk::SampleCountFlagBits::e1;
  vk::raii::Image colorImage = nullptr;
  vk::raii::DeviceMemory colorImageMemory = nullptr;
  vk::raii::ImageView colorImageView = nullptr;

  // Depth
  vk::raii::Image depthImage = nullptr;
  vk::raii::DeviceMemory depthImageMemory = nullptr;
  vk::raii::ImageView depthImageView = nullptr;

  // Texture
  uint32_t mipLevels = 1;
  vk::raii::Image textureImage = nullptr;
  vk::raii::DeviceMemory textureImageMemory = nullptr;
  vk::raii::ImageView textureImageView = nullptr;
  vk::raii::Sampler textureSampler = nullptr;

  // Vertex and Index Buffers
  std::vector<Vertex> vertices;
  std::vector<uint32_t> indices;
  vk::raii::Buffer vertexBuffer = nullptr;
  vk::raii::DeviceMemory vertexBufferMemory = nullptr;
  vk::raii::Buffer indexBuffer = nullptr;
  vk::raii::DeviceMemory indexBufferMemory = nullptr;

  // UBO
  std::vector<vk::raii::Buffer> uniformBuffers;
  std::vector<vk::raii::DeviceMemory> uniformBuffersMemory;
  std::vector<void *> uniformBuffersMapped;

  // Descriptors
  vk::raii::DescriptorPool descriptorPool = nullptr;
  std::vector<vk::raii::DescriptorSet> descriptorSets;

  // Commands
  vk::raii::CommandPool commandPool = nullptr;
  std::vector<vk::raii::CommandBuffer> commandBuffers;

  // Semaphores and Fences
  std::vector<vk::raii::Semaphore> renderFinishedSemaphores;
  std::vector<vk::raii::Fence> inFlightFences;
  std::vector<vk::raii::Semaphore> presentCompleteSemaphores;
  uint32_t frameIndex = 0;

  AppInfo appInfo = {};

  struct SwapChainSupportDetails {
    vk::SurfaceCapabilitiesKHR capabilities;
    std::vector<vk::SurfaceFormatKHR> formats;
    std::vector<vk::PresentModeKHR> presentModes;
  };

  // Resize
  bool framebufferResized = false;

  std::vector<const char *> requiredDeviceExtension = {
      vk::KHRPortabilitySubsetExtensionName,
      vk::KHRSwapchainExtensionName,
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
    // Setup
    createInstance();
    setupDebugMessenger();
    createSurface();
    pickPhysicalDevice();
    checkFeatureSupport();
    createLogicalDevice();
    createSwapChain();
    createImageViews();

    // Create render pass only if not using dynamic rendering
    if (!appInfo.profileSupported) {
      createRenderPass();
    }
    // Layout and Pipeline
    createDescriptorSetLayout();
    createGraphicsPipeline();

    createColorResources();
    createDepthResources();

    // Create framebuffers only if not using dynamic rendering
    if (!appInfo.profileSupported) {
      createFramebuffers();
    }

    // CommandPool and resources
    createCommandPool();

    // Texture
    createTextureImage();
    createTextureImageView();
    createTextureSampler();

    // Models
    loadModel();

    // Vertex and Index Buffer
    createVertexBuffer();
    createIndexBuffer();

    // UBO
    createUniformBuffers();

    // Descriptors
    createDescriptorPool();
    createDescriptorSets();

    // CommandBuffers
    createCommandBuffers();

    // Semaphores and Fences
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

    swapChainFramebuffers.clear();
    swapChainImageViews.clear();

    // Semaphores tied to swapchain image indices need to be rebuilt on resize
    presentCompleteSemaphores.clear();
    renderFinishedSemaphores.clear();
    for (auto &imageView : swapChainImageViews) {
      imageView = nullptr;
    }

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

    createColorResources();
    createDepthResources();

    if (!appInfo.profileSupported) {
      createRenderPass();
      createFramebuffers();
    }

    // Recreate per-swapchain-image present semaphores after resize
    presentCompleteSemaphores.reserve(swapChainImages.size());
    renderFinishedSemaphores.reserve(swapChainImages.size());
    vk::SemaphoreCreateInfo semaphoreInfo{};
    for (size_t i = 0; i < swapChainImages.size(); ++i) {
      presentCompleteSemaphores.push_back(
          device.createSemaphore(semaphoreInfo));
      renderFinishedSemaphores.push_back(device.createSemaphore(semaphoreInfo));
    }
  }

  void createInstance() {
    vk::ApplicationInfo appInfo{};
    appInfo.pApplicationName = "Hello Triangle";
    appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.pEngineName = "No Engine";
    appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
    appInfo.apiVersion = vk::ApiVersion14;

    // Get the required extensions.
    auto extensions = getRequiredInstanceExtensions();
    // MACOS ONLY MoltenVK portability flag
    extensions.push_back(vk::KHRPortabilityEnumerationExtensionName);

    vk::InstanceCreateInfo createInfo{};
    createInfo.pApplicationInfo = &appInfo;
    // Add the portability enumeration flag to enable MoltenVK support on macOS
    createInfo.flags = vk::InstanceCreateFlagBits::eEnumeratePortabilityKHR;
    createInfo.enabledExtensionCount = static_cast<uint32_t>(extensions.size());
    createInfo.ppEnabledExtensionNames = extensions.data();

    instance = vk::raii::Instance(context, createInfo);
  }

  void setupDebugMessenger() {
    // Always set up the debug messenger
    // It will only be used if validation layers are enabled via vulkanconfig

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

    try {
      debugMessenger = instance.createDebugUtilsMessengerEXT(
          debugUtilsMessengerCreateInfoEXT);
    } catch (vk::SystemError &err) {
      // If the debug utils extension is not available, this will fail
      // That's okay, it just means validation layers aren't enabled
      std::cout << "Debug messenger not available. Validation layers may not "
                   "be enabled."
                << std::endl;
    }
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
        physicalDevice.getProperties().apiVersion >= VK_API_VERSION_1_3;

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

    // Return true if the physicalDevice meets all the criteria
    return supportsVulkan1_3 && supportsGraphics &&
           supportsAllRequiredExtensions;
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
    msaaSamples = getMaxUsableSampleCount();

    // Print device information
    vk::PhysicalDeviceProperties deviceProperties =
        physicalDevice.getProperties();
    std::cout << "Selected GPU: " << deviceProperties.deviceName << std::endl;
    std::cout << "API Version: "
              << VK_VERSION_MAJOR(deviceProperties.apiVersion) << "."
              << VK_VERSION_MINOR(deviceProperties.apiVersion) << "."
              << VK_VERSION_PATCH(deviceProperties.apiVersion) << std::endl;
  }

  void checkFeatureSupport() {
    // Define the KHR roadmap 2022 profile - more widely supported than 2024
    appInfo.profile = {VP_MY_PROFILE_NAME, VP_MY_PROFILE_SPEC_VERSION};

    // Check if the profile is supported
    VkBool32 supported = vk::False;
    VkResult result = vpGetPhysicalDeviceProfileSupport(
        *instance, *physicalDevice, &appInfo.profile, &supported);

    if (result == VK_SUCCESS && supported == vk::True) {
      appInfo.profileSupported = true;
      std::cout << "Using custom profile" << std::endl;
    }

    else {
      appInfo.profileSupported = false;
      std::cout
          << "Falling back to traditional rendering (profile not supported)"
          << std::endl;
    }

    // If we wanted to implement fallback, we would call detectFeatureSupport()
    // here But for this example, we'll just use traditional rendering if the
    // profile isn't supported
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

    float queuePriority = 0.5f;
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo{};
    deviceQueueCreateInfo.queueFamilyIndex = queueIndex;
    deviceQueueCreateInfo.queueCount = 1;
    deviceQueueCreateInfo.pQueuePriorities = &queuePriority;

    if (appInfo.profileSupported) {

      // Create device with Best Practices Profile

      // Enable required features
      vk::PhysicalDeviceFeatures2 features2;
      vk::PhysicalDeviceFeatures deviceFeatures{};
      deviceFeatures.samplerAnisotropy = vk::True;
      deviceFeatures.sampleRateShading = vk::True;
      features2.features = deviceFeatures;

      // Enable dynamic rendering
      vk::PhysicalDeviceDynamicRenderingFeatures dynamicRenderingFeatures;
      dynamicRenderingFeatures.dynamicRendering = vk::True;

      // Enable synchronization2
      vk::PhysicalDeviceSynchronization2Features synchronization2Features;
      synchronization2Features.synchronization2 = vk::True;

      // Chain them together: features2 -> dynamicRenderingFeatures ->
      // synchronization2Features
      features2.pNext = &dynamicRenderingFeatures;
      dynamicRenderingFeatures.pNext = &synchronization2Features;

      // Create a vk::DeviceCreateInfo with the required features
      vk::DeviceCreateInfo deviceCreateInfo{};
      deviceCreateInfo.pNext = &features2;
      deviceCreateInfo.queueCreateInfoCount = 1;
      deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
      deviceCreateInfo.enabledExtensionCount =
          static_cast<uint32_t>(requiredDeviceExtension.size());
      deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();
      // Create the device with the vk::DeviceCreateInfo
      device = vk::raii::Device(physicalDevice, deviceCreateInfo);

      std::cout << " Created logical device using custom profile" << std::endl;
    } else {
      // Fallback to manual device creation
      // Enable required features
      vk::PhysicalDeviceFeatures2 features2;
      vk::PhysicalDeviceFeatures deviceFeatures{};
      deviceFeatures.samplerAnisotropy = vk::True;
      deviceFeatures.sampleRateShading = vk::True;
      features2.features = deviceFeatures;

      // Incomplete fallback since still using PipelineBarrier2 that requires
      // synchronization2 Enable synchronization2
      vk::PhysicalDeviceSynchronization2Features synchronization2Features;
      synchronization2Features.synchronization2 = vk::True;
      features2.pNext = &synchronization2Features;

      vk::DeviceCreateInfo deviceCreateInfo{};
      deviceCreateInfo.queueCreateInfoCount = 1;
      deviceCreateInfo.pQueueCreateInfos = &deviceQueueCreateInfo;
      deviceCreateInfo.enabledExtensionCount =
          static_cast<uint32_t>(requiredDeviceExtension.size());
      deviceCreateInfo.ppEnabledExtensionNames = requiredDeviceExtension.data();
      deviceCreateInfo.pNext = &features2;

      device = vk::raii::Device(physicalDevice, deviceCreateInfo);

      std::cout << " Created logical device using manual feature selection"
                << std::endl;
    }
    queue = device.getQueue(queueIndex, 0);
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
                          vk::ImageAspectFlagBits::eColor, 1));
    }
  }

  void createRenderPass() {
    // This is only called if the Best Practices profile is not supported
    // or if dynamic rendering is not available
    vk::AttachmentDescription colorAttachment{};
    colorAttachment.format = swapChainSurfaceFormat.format;
    colorAttachment.samples = msaaSamples;
    colorAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    colorAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachment.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachment.finalLayout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentDescription depthAttachment{};
    depthAttachment.format = findDepthFormat(),
    depthAttachment.samples = msaaSamples,
    depthAttachment.loadOp = vk::AttachmentLoadOp::eClear;
    depthAttachment.storeOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    depthAttachment.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    depthAttachment.initialLayout = vk::ImageLayout::eUndefined;
    depthAttachment.finalLayout =
        vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentDescription colorAttachmentResolve{};
    colorAttachmentResolve.format = swapChainSurfaceFormat.format;
    colorAttachmentResolve.samples = vk::SampleCountFlagBits::e1;
    colorAttachmentResolve.loadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachmentResolve.storeOp = vk::AttachmentStoreOp::eStore;
    colorAttachmentResolve.stencilLoadOp = vk::AttachmentLoadOp::eDontCare;
    colorAttachmentResolve.stencilStoreOp = vk::AttachmentStoreOp::eDontCare;
    colorAttachmentResolve.initialLayout = vk::ImageLayout::eUndefined;
    colorAttachmentResolve.finalLayout = vk::ImageLayout::ePresentSrcKHR;

    vk::AttachmentReference colorAttachmentRef{};
    colorAttachmentRef.attachment = 0;
    colorAttachmentRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::AttachmentReference depthAttachmentRef{};
    depthAttachmentRef.attachment = 1;
    depthAttachmentRef.layout = vk::ImageLayout::eDepthStencilAttachmentOptimal;

    vk::AttachmentReference colorAttachmentResolveRef{};
    colorAttachmentResolveRef.attachment = 2,
    colorAttachmentResolveRef.layout = vk::ImageLayout::eColorAttachmentOptimal;

    vk::SubpassDescription subpass{};
    subpass.pipelineBindPoint = vk::PipelineBindPoint::eGraphics;
    subpass.colorAttachmentCount = 1;
    subpass.pColorAttachments = &colorAttachmentRef;
    subpass.pResolveAttachments = &colorAttachmentResolveRef;
    subpass.pDepthStencilAttachment = &depthAttachmentRef;

    vk::SubpassDependency dependency{};
    dependency.srcSubpass = VK_SUBPASS_EXTERNAL;
    dependency.dstSubpass = 0;
    dependency.srcStageMask =
        vk::PipelineStageFlagBits::eColorAttachmentOutput |
        vk::PipelineStageFlagBits::eEarlyFragmentTests;
    dependency.dstStageMask =
        vk::PipelineStageFlagBits::eColorAttachmentOutput |
        vk::PipelineStageFlagBits::eEarlyFragmentTests;
    dependency.srcAccessMask = vk::AccessFlagBits::eNone;
    dependency.dstAccessMask = vk::AccessFlagBits::eColorAttachmentWrite |
                               vk::AccessFlagBits::eDepthStencilAttachmentWrite;

    std::array<vk::AttachmentDescription, 3> attachments = {
        colorAttachment, depthAttachment, colorAttachmentResolve};
    vk::RenderPassCreateInfo renderPassInfo{};
    renderPassInfo.attachmentCount = static_cast<uint32_t>(attachments.size());
    renderPassInfo.pAttachments = attachments.data();
    renderPassInfo.subpassCount = 1;
    renderPassInfo.pSubpasses = &subpass;
    renderPassInfo.dependencyCount = 1;
    renderPassInfo.pDependencies = &dependency;

    renderPass = device.createRenderPass(renderPassInfo);
  }

  void createDescriptorSetLayout() {
    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{};

    // uniform buffer object descriptor set layout binding for the vertex shader
    bindings[0].binding = 0;
    bindings[0].descriptorType = vk::DescriptorType::eUniformBuffer;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = vk::ShaderStageFlagBits::eVertex;

    // combined image sampler descriptor set layout binding for the texture
    bindings[1].binding = 1;
    bindings[1].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = vk::ShaderStageFlagBits::eFragment;

    vk::DescriptorSetLayoutCreateInfo layoutInfo{};
    layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
    layoutInfo.pBindings = bindings.data();
    descriptorSetLayout = vk::raii::DescriptorSetLayout(device, layoutInfo);
  }

  void createGraphicsPipeline() {
    vk::raii::ShaderModule shaderModule =
        createShaderModule(readFile("simple.spv"));

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
    inputAssembly.primitiveRestartEnable = vk::False;

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
    multisampling.rasterizationSamples = msaaSamples;
    multisampling.sampleShadingEnable = vk::True;
    multisampling.minSampleShading = 0.2f;

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

    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{};
    pipelineLayoutInfo.setLayoutCount = 1;
    pipelineLayoutInfo.pSetLayouts = &*descriptorSetLayout;
    pipelineLayoutInfo.pushConstantRangeCount = 0;
    pipelineLayout = device.createPipelineLayout(pipelineLayoutInfo);

    // Configure pipeline based on whether we're using the KHR roadmap 2022
    // profile With the KHR roadmap 2022 profile, we can use dynamic rendering
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

    if (appInfo.profileSupported) {
      std::cout << "Creating pipeline with dynamic rendering (custom profile)"
                << std::endl;
    } else {
      std::cout << "Creating pipeline with traditional renderpass (fallback)"
                << std::endl;
      pipelineCreateInfoChain.unlink<vk::PipelineRenderingCreateInfo>();
      pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>().renderPass =
          *renderPass;
    }

    graphicsPipeline = vk::raii::Pipeline(
        device, nullptr,
        pipelineCreateInfoChain.get<vk::GraphicsPipelineCreateInfo>());
  }

  void createFramebuffers() {
    // This is only called if the Best Practices profile is not supported
    // or if dynamic rendering is not available
    swapChainFramebuffers.reserve(swapChainImageViews.size());

    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
      std::array<vk::ImageView, 3> attachments = {
          *colorImageView, *depthImageView, *swapChainImageViews[i]};

      vk::FramebufferCreateInfo framebufferInfo{};
      framebufferInfo.renderPass = *renderPass;
      framebufferInfo.attachmentCount =
          static_cast<uint32_t>(attachments.size());
      framebufferInfo.pAttachments = attachments.data();
      framebufferInfo.width = swapChainExtent.width;
      framebufferInfo.height = swapChainExtent.height;
      framebufferInfo.layers = 1;

      swapChainFramebuffers.push_back(
          device.createFramebuffer(framebufferInfo));
    }
  }

  void createCommandPool() {
    vk::CommandPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::CommandPoolCreateFlagBits::eResetCommandBuffer;
    poolInfo.queueFamilyIndex = queueIndex;
    commandPool = vk::raii::CommandPool(device, poolInfo);
  }

  void createColorResources() {
    vk::Format colorFormat = swapChainSurfaceFormat.format;

    std::tie(colorImage, colorImageMemory) =
        createImage(swapChainExtent.width, swapChainExtent.height, 1,
                    msaaSamples, colorFormat, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransientAttachment |
                        vk::ImageUsageFlagBits::eColorAttachment,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
    colorImageView = createImageView(colorImage, colorFormat,
                                     vk::ImageAspectFlagBits::eColor, 1);
  }

  void createDepthResources() {
    vk::Format depthFormat = findDepthFormat();
    std::tie(depthImage, depthImageMemory) =
        createImage(swapChainExtent.width, swapChainExtent.height, 1,
                    msaaSamples, depthFormat, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eDepthStencilAttachment,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);
    depthImageView = createImageView(depthImage, depthFormat,
                                     vk::ImageAspectFlagBits::eDepth, 1);
  }

  vk::Format findSupportedFormat(const std::vector<vk::Format> candidates,
                                 vk::ImageTiling tiling,
                                 vk::FormatFeatureFlagBits features) {
    for (const auto format : candidates) {
      vk::FormatProperties props = physicalDevice.getFormatProperties(format);

      if (tiling == vk::ImageTiling::eLinear &&
          (props.linearTilingFeatures & features) == features) {
        return format;
      }
      if (tiling == vk::ImageTiling::eOptimal &&
          (props.optimalTilingFeatures & features) == features) {
        return format;
      }
    }

    throw std::runtime_error("failed to find supported format!");
  }

  [[nodiscard]] vk::Format findDepthFormat() {
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

  void createTextureImage() {
    // Load KTX texture instead of using stb_image
    ktxTexture *kTexture;
    KTX_error_code result = ktxTexture_CreateFromNamedFile(
        TEXTURE_PATH.c_str(), KTX_TEXTURE_CREATE_LOAD_IMAGE_DATA_BIT,
        &kTexture);

    if (result != KTX_SUCCESS) {
      throw std::runtime_error("failed to load texture image!");
    }

    // Get texture dimensions and data
    uint32_t texWidth = kTexture->baseWidth;
    uint32_t texHeight = kTexture->baseHeight;
    ktx_size_t imageSize = ktxTexture_GetImageSize(kTexture, 0);
    ktx_uint8_t *ktxTextureData = ktxTexture_GetData(kTexture);

    // Create staging buffer and copy texture data to it
    auto [stagingBuffer, stagingBufferMemory] =
        createBuffer(imageSize, vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible |
                         vk::MemoryPropertyFlagBits::eHostCoherent);
    void *data = stagingBufferMemory.mapMemory(0, imageSize);
    std::memcpy(data, ktxTextureData, imageSize);
    stagingBufferMemory.unmapMemory();

    // Determine the Vulkan format from KTX format
    vk::Format textureFormat =
        vk::Format::eR8G8B8A8Srgb; // Default format, should be determined from
                                   // KTX metadata

    // Create the texture image
    std::tie(textureImage, textureImageMemory) =
        createImage(texWidth, texHeight, mipLevels, vk::SampleCountFlagBits::e1,
                    textureFormat, vk::ImageTiling::eOptimal,
                    vk::ImageUsageFlagBits::eTransferSrc |
                        vk::ImageUsageFlagBits::eTransferDst |
                        vk::ImageUsageFlagBits::eSampled,
                    vk::MemoryPropertyFlagBits::eDeviceLocal);

    // Copy data from staging buffer to texture image and generate mipmaps
    transitionImageLayout(textureImage, vk::ImageLayout::eUndefined,
                          vk::ImageLayout::eTransferDstOptimal, mipLevels);
    copyBufferToImage(stagingBuffer, textureImage,
                      static_cast<uint32_t>(texWidth),
                      static_cast<uint32_t>(texHeight));
    generateMipmaps(textureImage, textureFormat, texWidth, texHeight,
                    mipLevels);

    // Clean up KTX resources
    ktxTexture_Destroy(kTexture);
  }

  void generateMipmaps(vk::raii::Image &image, vk::Format imageFormat,
                       int32_t texWidth, int32_t texHeight,
                       uint32_t mipLevels) {
    // Check if image format supports linear blit-ing
    vk::FormatProperties formatProperties =
        physicalDevice.getFormatProperties(imageFormat);

    if (!(formatProperties.optimalTilingFeatures &
          vk::FormatFeatureFlagBits::eSampledImageFilterLinear)) {
      throw std::runtime_error(
          "texture image format does not support linear blitting!");
    }

    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier{};
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.baseArrayLayer = 0;
    barrier.subresourceRange.layerCount = 1;
    barrier.subresourceRange.levelCount = 1;

    int32_t mipWidth = texWidth;
    int32_t mipHeight = texHeight;

    for (uint32_t i = 1; i < mipLevels; i++) {
      barrier.subresourceRange.baseMipLevel = i - 1;
      barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
      barrier.newLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
      barrier.dstAccessMask = vk::AccessFlagBits::eTransferRead;

      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eTransfer, {},
                                    {}, {}, barrier);

      vk::ArrayWrapper1D<vk::Offset3D, 2> offsets, dstOffsets;
      offsets[0] = vk::Offset3D{0, 0, 0};
      offsets[1] = vk::Offset3D{mipWidth, mipHeight, 1};
      dstOffsets[0] = vk::Offset3D{0, 0, 0};
      dstOffsets[1] = vk::Offset3D{mipWidth > 1 ? mipWidth / 2 : 1,
                                   mipHeight > 1 ? mipHeight / 2 : 1, 1};
      vk::ImageBlit blit{};
      blit.srcSubresource = vk::ImageSubresourceLayers{
          vk::ImageAspectFlagBits::eColor, i - 1, 0, 1};
      blit.dstSubresource =
          vk::ImageSubresourceLayers{vk::ImageAspectFlagBits::eColor, i, 0, 1};
      blit.srcOffsets = offsets;
      blit.dstOffsets = dstOffsets;

      commandBuffer.blitImage(image, vk::ImageLayout::eTransferSrcOptimal,
                              image, vk::ImageLayout::eTransferDstOptimal,
                              {blit}, vk::Filter::eLinear);

      barrier.oldLayout = vk::ImageLayout::eTransferSrcOptimal;
      barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
      barrier.srcAccessMask = vk::AccessFlagBits::eTransferRead;
      barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

      commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                    vk::PipelineStageFlagBits::eFragmentShader,
                                    {}, {}, {}, barrier);

      if (mipWidth > 1)
        mipWidth /= 2;
      if (mipHeight > 1)
        mipHeight /= 2;
    }

    barrier.subresourceRange.baseMipLevel = mipLevels - 1;
    barrier.oldLayout = vk::ImageLayout::eTransferDstOptimal;
    barrier.newLayout = vk::ImageLayout::eShaderReadOnlyOptimal;
    barrier.srcAccessMask = vk::AccessFlagBits::eTransferWrite;
    barrier.dstAccessMask = vk::AccessFlagBits::eShaderRead;

    commandBuffer.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer,
                                  vk::PipelineStageFlagBits::eFragmentShader,
                                  {}, {}, {}, barrier);

    endSingleTimeCommands(commandBuffer);
  }

  vk::SampleCountFlagBits getMaxUsableSampleCount() {
    vk::PhysicalDeviceProperties physicalDeviceProperties =
        physicalDevice.getProperties();
    vk::SampleCountFlags counts =
        physicalDeviceProperties.limits.framebufferColorSampleCounts &
        physicalDeviceProperties.limits.framebufferDepthSampleCounts;
    if (counts & vk::SampleCountFlagBits::e64)
      return vk::SampleCountFlagBits::e64;
    if (counts & vk::SampleCountFlagBits::e32)
      return vk::SampleCountFlagBits::e32;
    if (counts & vk::SampleCountFlagBits::e16)
      return vk::SampleCountFlagBits::e16;
    if (counts & vk::SampleCountFlagBits::e8)
      return vk::SampleCountFlagBits::e8;
    if (counts & vk::SampleCountFlagBits::e4)
      return vk::SampleCountFlagBits::e4;
    if (counts & vk::SampleCountFlagBits::e2)
      return vk::SampleCountFlagBits::e2;

    return vk::SampleCountFlagBits::e1;
  }

  void createTextureImageView() {
    textureImageView =
        createImageView(*textureImage, vk::Format::eR8G8B8A8Srgb,
                        vk::ImageAspectFlagBits::eColor, mipLevels);
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
    samplerInfo.minLod = 0.0f;
    samplerInfo.maxLod = vk::LodClampNone;

    textureSampler = vk::raii::Sampler(device, samplerInfo);
  }

  [[nodiscard]] vk::raii::ImageView
  createImageView(vk::Image const &image, vk::Format format,
                  vk::ImageAspectFlags aspectFlags, uint32_t mipLevels) const {
    vk::ImageViewCreateInfo imageViewCreateInfo{};
    imageViewCreateInfo.image = image;
    imageViewCreateInfo.viewType = vk::ImageViewType::e2D;
    imageViewCreateInfo.format = format;
    imageViewCreateInfo.subresourceRange = {aspectFlags, 0, mipLevels, 0, 1};
    return vk::raii::ImageView(device, imageViewCreateInfo);
  }

  std::pair<vk::raii::Image, vk::raii::DeviceMemory>
  createImage(uint32_t width, uint32_t height, uint32_t mipLevels,
              vk::SampleCountFlagBits numSamples, vk::Format format,
              vk::ImageTiling tiling, vk::ImageUsageFlags usage,
              vk::MemoryPropertyFlags properties) {
    vk::ImageCreateInfo imageInfo{};
    imageInfo.imageType = vk::ImageType::e2D;
    imageInfo.format = format;
    imageInfo.extent = vk::Extent3D{width, height, 1};
    imageInfo.mipLevels = mipLevels;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = numSamples;
    imageInfo.tiling = tiling;
    imageInfo.usage = usage;
    imageInfo.sharingMode = vk::SharingMode::eExclusive;
    imageInfo.initialLayout = vk::ImageLayout::eUndefined;

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
                             vk::ImageLayout newLayout, uint32_t mipLevels) {
    const auto commandBuffer = beginSingleTimeCommands();

    vk::ImageMemoryBarrier barrier{};
    barrier.oldLayout = oldLayout;
    barrier.newLayout = newLayout;
    barrier.srcQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.dstQueueFamilyIndex = vk::QueueFamilyIgnored;
    barrier.image = image;
    barrier.subresourceRange.aspectMask = vk::ImageAspectFlagBits::eColor;
    barrier.subresourceRange.levelCount = mipLevels;
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
    vk::raii::CommandBuffer commandBuffer = beginSingleTimeCommands();

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
    fastgltf::Parser parser;

    auto data = fastgltf::GltfDataBuffer::FromPath(MODEL_PATH);
    if (data.error() != fastgltf::Error::None)
      throw std::runtime_error("Failed to read glTF file: " +
                               (MODEL_PATH).string());

    auto assetResult =
        parser.loadGltfBinary(data.get(), MODEL_PATH.parent_path(),
                              fastgltf::Options::LoadExternalBuffers |
                                  fastgltf::Options::GenerateMeshIndices);

    if (assetResult.error() != fastgltf::Error::None)
      throw std::runtime_error(
          "Failed to parse glTF: " +
          std::string(fastgltf::getErrorMessage(assetResult.error())));

    fastgltf::Asset &model = assetResult.get();

    vertices.clear();
    indices.clear();

    for (const auto &mesh : model.meshes) {
      for (const auto &primitive : mesh.primitives) {
        uint32_t baseVertex = static_cast<uint32_t>(vertices.size());

        // --- Positions (required) ---
        auto posAttr = primitive.findAttribute("POSITION");
        if (posAttr == primitive.attributes.end())
          throw std::runtime_error("glTF primitive missing POSITION attribute");

        auto &posAccessor = model.accessors[posAttr->accessorIndex];
        vertices.resize(vertices.size() + posAccessor.count);

        fastgltf::iterateAccessorWithIndex<glm::vec3>(
            model, posAccessor, [&](glm::vec3 pos, size_t idx) {
              // glTF: right-handed, Y-up. Vulkan clip space: Y-down.
              Vertex &v = vertices[baseVertex + idx];
              v.pos = glm::vec3(pos.x, -pos.y, pos.z);
              v.color = glm::vec3(1.0f);
              v.texCoord = glm::vec2(0.0f);
            });

        // --- UVs (optional) ---
        if (auto uvAttr = primitive.findAttribute("TEXCOORD_0");
            uvAttr != primitive.attributes.end()) {
          auto &uvAccessor = model.accessors[uvAttr->accessorIndex];
          fastgltf::iterateAccessorWithIndex<glm::vec2>(
              model, uvAccessor, [&](glm::vec2 uv, size_t idx) {
                vertices[baseVertex + idx].texCoord = uv;
              });
        }

        // --- Indices (required) ---
        if (!primitive.indicesAccessor.has_value())
          throw std::runtime_error("glTF primitive missing indices");

        auto &indexAccessor =
            model.accessors[primitive.indicesAccessor.value()];
        indices.reserve(indices.size() + indexAccessor.count);

        // fastgltf transparently widens ubyte/ushort/uint index accessors to
        // uint32_t here.
        fastgltf::iterateAccessor<uint32_t>(
            model, indexAccessor,
            [&](uint32_t index) { indices.push_back(baseVertex + index); });
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

  void createDescriptorPool() {
    std::array<vk::DescriptorPoolSize, 2> poolSize{};

    poolSize[0].type = vk::DescriptorType::eUniformBuffer;
    poolSize[0].descriptorCount = MAX_FRAMES_IN_FLIGHT;

    poolSize[1].type = vk::DescriptorType::eCombinedImageSampler;
    poolSize[1].descriptorCount = MAX_FRAMES_IN_FLIGHT;

    vk::DescriptorPoolCreateInfo poolInfo{};
    poolInfo.flags = vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet;
    poolInfo.maxSets = MAX_FRAMES_IN_FLIGHT;
    poolInfo.poolSizeCount = static_cast<uint32_t>(poolSize.size());
    poolInfo.pPoolSizes = poolSize.data();

    descriptorPool = vk::raii::DescriptorPool(device, poolInfo);
  }

  void createDescriptorSets() {
    std::vector<vk::DescriptorSetLayout> layouts(MAX_FRAMES_IN_FLIGHT,
                                                 *descriptorSetLayout);
    vk::DescriptorSetAllocateInfo allocInfo{};
    allocInfo.descriptorPool = *descriptorPool;
    allocInfo.descriptorSetCount = static_cast<uint32_t>(layouts.size());
    allocInfo.pSetLayouts = layouts.data();

    descriptorSets.clear();
    descriptorSets = device.allocateDescriptorSets(allocInfo);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      vk::DescriptorBufferInfo bufferInfo{};
      bufferInfo.buffer = uniformBuffers[i];
      bufferInfo.offset = 0;
      bufferInfo.range = sizeof(UniformBufferObject);

      vk::DescriptorImageInfo imageInfo{};
      imageInfo.sampler = textureSampler;
      imageInfo.imageView = textureImageView;
      imageInfo.imageLayout = vk::ImageLayout::eShaderReadOnlyOptimal;

      std::array<vk::WriteDescriptorSet, 2> descriptorWrites{};
      descriptorWrites[0].dstSet = descriptorSets[i];
      descriptorWrites[0].dstBinding = 0;
      descriptorWrites[0].dstArrayElement = 0;
      descriptorWrites[0].descriptorCount = 1;
      descriptorWrites[0].descriptorType = vk::DescriptorType::eUniformBuffer;
      descriptorWrites[0].pBufferInfo = &bufferInfo;

      descriptorWrites[1].dstSet = descriptorSets[i];
      descriptorWrites[1].dstBinding = 1;
      descriptorWrites[1].dstArrayElement = 0;
      descriptorWrites[1].descriptorCount = 1;
      descriptorWrites[1].descriptorType =
          vk::DescriptorType::eCombinedImageSampler;
      descriptorWrites[1].pImageInfo = &imageInfo;

      device.updateDescriptorSets(descriptorWrites, {});
    }
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
    endSingleTimeCommands(std::move(commandCopyBuffer));
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
    vk::CommandBufferAllocateInfo allocInfo{};
    allocInfo.commandPool = commandPool;
    allocInfo.level = vk::CommandBufferLevel::ePrimary;
    allocInfo.commandBufferCount = MAX_FRAMES_IN_FLIGHT;
    commandBuffers = vk::raii::CommandBuffers(device, allocInfo);
  }

  void recordCommandBuffer(uint32_t imageIndex) {
    auto &commandBuffer = commandBuffers[frameIndex];
    commandBuffer.begin({});

    // Transition the attachments to the correct layouts for dynamic rendering

    // Before starting rendering, transition the swapchain image to
    // vk::ImageLayout::eColorAttachmentOptimal 1) Multisampled color attachment
    // image -> ColorAttachmentOptimal

    // Transition the multisampled color image to color attachment optimal
    // layout
    transition_image_layout(*colorImage, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eColorAttachmentOptimal,
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            vk::AccessFlagBits2::eColorAttachmentWrite,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::PipelineStageFlagBits2::eColorAttachmentOutput,
                            vk::ImageAspectFlagBits::eColor);

    // 2) Depth attachment image -> DepthStencilAttachmentOptimal
    transition_image_layout(*depthImage, vk::ImageLayout::eUndefined,
                            vk::ImageLayout::eDepthAttachmentOptimal,
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                            vk::AccessFlagBits2::eDepthStencilAttachmentWrite,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits2::eLateFragmentTests,
                            vk::PipelineStageFlagBits2::eEarlyFragmentTests |
                                vk::PipelineStageFlagBits2::eLateFragmentTests,
                            vk::ImageAspectFlagBits::eDepth);

    // 3) Resolve (swapchain) image -> ColorAttachmentOptimal
    transition_image_layout(
        swapChainImages[imageIndex], vk::ImageLayout::eUndefined,
        vk::ImageLayout::eColorAttachmentOptimal,
        {}, // srcAccessMask (no need to wait for previous operations)
        vk::AccessFlagBits2::eColorAttachmentWrite,         // dstAccessMask
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // srcStage
        vk::PipelineStageFlagBits2::eColorAttachmentOutput, // dstStage
        vk::ImageAspectFlagBits::eColor);

    // Clear values for color and depth
    vk::ClearValue clearColor =
        vk::ClearColorValue(std::array<float, 4>{0.0f, 0.0f, 0.0f, 1.0f});
    vk::ClearValue clearDepth = vk::ClearDepthStencilValue(1.0f, 0);
    std::array<vk::ClearValue, 2> clearValues = {clearColor, clearDepth};

    // Use different rendering approach based on profile support
    if (appInfo.profileSupported) {
      // Use dynamic rendering with the KHR roadmap 2022 profile
      vk::RenderingAttachmentInfo colorAttachmentInfo = {};
      colorAttachmentInfo.imageView = colorImageView;
      colorAttachmentInfo.imageLayout =
          vk::ImageLayout::eColorAttachmentOptimal;
      colorAttachmentInfo.resolveImageView = swapChainImageViews[imageIndex];
      colorAttachmentInfo.resolveImageLayout =
          vk::ImageLayout::eColorAttachmentOptimal;
      colorAttachmentInfo.resolveMode = vk::ResolveModeFlagBits::eAverage;
      colorAttachmentInfo.loadOp = vk::AttachmentLoadOp::eClear;
      colorAttachmentInfo.storeOp = vk::AttachmentStoreOp::eStore;
      colorAttachmentInfo.clearValue = clearColor;

      vk::RenderingAttachmentInfo depthAttachmentInfo = {};
      depthAttachmentInfo.imageView = depthImageView;
      depthAttachmentInfo.imageLayout =
          vk::ImageLayout::eDepthAttachmentOptimal;
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
    } else {
      // Use traditional render pass if not using the KHR roadmap 2022 profile
      vk::RenderPassBeginInfo renderPassInfo{};
      renderPassInfo.renderPass = *renderPass;
      renderPassInfo.framebuffer = *swapChainFramebuffers[imageIndex];
      renderPassInfo.renderArea.offset = {.x = 0, .y = 0};
      renderPassInfo.renderArea.extent = swapChainExtent;
      renderPassInfo.clearValueCount =
          static_cast<uint32_t>(clearValues.size());
      renderPassInfo.pClearValues = clearValues.data();

      commandBuffer.beginRenderPass(renderPassInfo,
                                    vk::SubpassContents::eInline);
    }

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
    commandBuffer.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,
                                     pipelineLayout, 0,
                                     *descriptorSets[frameIndex], nullptr);
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0,
                              0);

    if (appInfo.profileSupported) {
      commandBuffer.endRendering();

      // After rendering, transition the swapchain image to
      // vk::ImageLayout::ePresentSrcKHR
      transition_image_layout(
          swapChainImages[imageIndex], vk::ImageLayout::eColorAttachmentOptimal,
          vk::ImageLayout::ePresentSrcKHR,
          vk::AccessFlagBits2::eColorAttachmentWrite, {},
          vk::PipelineStageFlagBits2::eColorAttachmentOutput,
          vk::PipelineStageFlagBits2::eBottomOfPipe,
          vk::ImageAspectFlagBits::eColor);
    } else {
      commandBuffer.endRenderPass();
      // Traditional render pass already transitions the image to the correct
      // layout
    }

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
    renderFinishedSemaphores.reserve(MAX_FRAMES_IN_FLIGHT);
    inFlightFences.reserve(MAX_FRAMES_IN_FLIGHT);
    presentCompleteSemaphores.reserve(swapChainImages.size());

    vk::SemaphoreCreateInfo semaphoreInfo{};
    vk::FenceCreateInfo fenceInfo(vk::FenceCreateFlagBits::eSignaled);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
      inFlightFences.push_back(device.createFence(fenceInfo));
    }
    for (size_t i = 0; i < swapChainImages.size(); i++) {
      renderFinishedSemaphores.push_back(device.createSemaphore(semaphoreInfo));
      presentCompleteSemaphores.push_back(
          device.createSemaphore(semaphoreInfo));
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

  void drawFrame() {

    // Note: inFlightFences, presentCompleteSemaphores, and commandBuffers are
    // indexed by frameIndex,
    //       while renderFinishedSemaphores is indexed by imageIndex
    auto fenceResult =
        device.waitForFences(*inFlightFences[frameIndex], vk::True, UINT64_MAX);
    if (fenceResult != vk::Result::eSuccess) {
      throw std::runtime_error("Failed to wait for draw fence!");
    }

    auto [result, imageIndex] = swapChain.acquireNextImage(
        UINT64_MAX, *presentCompleteSemaphores[frameIndex], nullptr);

    // Due to VULKAN_HPP_HANDLE_ERROR_OUT_OF_DATE_AS_SUCCESS being defined,
    // eErrorOutOfDateKHR can be checked as a result here and does not need to
    // be caught by an exception.
    if (result == vk::Result::eErrorOutOfDateKHR) {
      recreateSwapChain();
      return;
    }

    // On other success codes than eSuccess and eSuboptimalKHR we just throw an
    // exception. On any error code, aquireNextImage already threw an exception.
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
    // eErrorOutOfDateKHR can be checked as a result here and does not need to
    // be caught by an exception.
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

    // Check if the debug utils extension is available
    std::vector<vk::ExtensionProperties> props =
        context.enumerateInstanceExtensionProperties();
    bool debugUtilsAvailable =
        std::ranges::any_of(props, [](vk::ExtensionProperties const &ep) {
          return strcmp(ep.extensionName, vk::EXTDebugUtilsExtensionName) == 0;
        });

    // Always include the debug utils extension if available
    // This allows validation layers to be enabled via vulkanconfig
    if (debugUtilsAvailable) {
      extensions.push_back(vk::EXTDebugUtilsExtensionName);
    } else {
      std::cout << "VK_EXT_debug_utils extension not available. Validation "
                   "layers may not work."
                << std::endl;
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
