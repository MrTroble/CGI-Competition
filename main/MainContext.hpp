#pragma once

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vulkan/vulkan.hpp>
#include <GLFW/glfw3.h>
#include <glm/glm.hpp>
#include "trackball.hpp"
#include "rng.hpp"
#include "Shared.hpp"

namespace fs = std::filesystem;


#define NUMBER_OF_BIG_LIGHTS 64
#define MAX_TILLING 8

#define MAX_IMAGE_TEXTURES_RTX 256

struct DataBuffer {
	vk::DeviceMemory memory;
	vk::Buffer buffer;
};

struct LightList {
	uint32_t ids[NUMBER_OF_LIGHTS];
	uint32_t amount;
	uint32_t pad1[15] = { 0 };
};

struct PerNodeData {
	glm::mat4 mvp;
	glm::mat4 previousViewMatrix;
	glm::mat4 inverseVP;
	glm::vec4 windowParameter;
	std::uint8_t padding[48];
	// TODO: add more things here
};

struct CameraData {
	glm::mat4 mvp;
	glm::mat4 previousViewMatrix;
	glm::mat4 inverseVP;
	glm::mat4 invView;
	glm::mat4 invProj;
	glm::vec4 windowParameter;
	uint32_t frame_Index;
	std::uint8_t padding[176];
	// TODO: add more things here
};

struct MaterialPushConst {
	uint32_t materialID;
	float alphaCutoff;
};

struct ArrayBuffer {
	vk::DescriptorPool pool;
	std::vector<vk::DescriptorSet> descriptors;
	vk::Buffer buffer;
	vk::DeviceMemory memory;
};

struct RenderTillingInfo {
	uint32_t xAmount = 1;
	uint32_t yAmount = 1;
	uint32_t zAmount = 1;

	uint32_t xOffset = 0;
	uint32_t yOffset = 0;
	uint32_t zOffset = 0;
};

struct Settings {
	bool showLights = false;
	bool useTAA = false;
	bool useAO = false;
	bool useShadowMaps = false;
	bool useRayTracing = false;	
	bool accumulate = false;

	RenderTillingInfo baseTillingInfo;
};



struct ShadowMapContext {
	std::vector<vk::Image> shadowMaps;
	std::vector<vk::ImageView> shadowMapViews;
	vk::DeviceMemory shadowMapMatrices;
};

struct RaytracingExtHolder {
	vk::Buffer buffer[2];
	vk::Buffer scratch[2];
	std::vector<vk::DeviceMemory> memory;
	vk::AccelerationStructureKHR accelerationStructures;
	vk::DescriptorSet accelerationSet;
	vk::AccelerationStructureInstanceKHR instanceInfo;
};

struct RaytracingMainContext {
	vk::Pipeline mainRayPipeline;
	vk::DescriptorSetLayout descLayout;
	vk::DescriptorSetLayout descImageLayout;
	vk::DescriptorSetLayout descTextureLayout;

	vk::PipelineLayout mainRayLayout;
	vk::DescriptorPool pool;
	vk::DescriptorPool imagePool;
	std::vector<vk::DescriptorSet> imageSet;
	vk::DescriptorSet textureSet;

	vk::AccelerationStructureKHR topLevelStructure;
	vk::DescriptorSet topLevelDescSet;

	vk::StridedDeviceAddressRegionKHR raygenRegion;
	vk::StridedDeviceAddressRegionKHR missRegion;
	vk::StridedDeviceAddressRegionKHR hitRegion;
	vk::StridedDeviceAddressRegionKHR callableRegion;

	DataBuffer geoBuffer;
	DataBuffer materialBuffer;
	DataBuffer instanceBuffer;
	std::array<DataBuffer, 3> additionalBuffer;
};

struct MainContext {
	vk::Instance instance;
	vk::Device device;
	vk::Queue queue;
	vk::PhysicalDevice usedPhyDevice;
	uint32_t queueIndex;
	GLFWwindow* window;
	vk::RenderPass renderPass;
	vk::SurfaceKHR surface;
	vk::SurfaceFormatKHR formatUsed;
	vk::Extent2D currentExtent;
	vk::SwapchainKHR swapchain;
	std::vector<vk::Image> swapchainImages;
	std::vector<vk::ImageView> swapchainImageViews;
	std::vector<vk::Framebuffer> framebuffer;
	vk::CommandPool commandPool;
	std::vector<vk::CommandBuffer> command;
	vk::Semaphore waitSemaphore;
	vk::Semaphore signalSemaphore;
	vk::Fence fence;
	vk::DescriptorPool descriptorPool;
	vk::detail::DispatchLoaderDynamic dynamicLoader;

	RaytracingMainContext raytraceExt;

	vk::DescriptorSetLayout modelLayout = nullptr;
	vk::DescriptorSetLayout cameraLayout = nullptr;
	vk::DescriptorSetLayout textureLayout = nullptr;

	vk::PipelineLayout pipelineLayout = nullptr;
	uint32_t deviceLocalOnly = UINT32_MAX;
	uint32_t hostVisibleOnly = UINT32_MAX;
	uint32_t hostVisibleDeviceLocalMemory = UINT32_MAX;

	std::vector<vk::ShaderModule> shaderModules;
	std::vector<vk::ShaderModule> shaderModulesRaytracing;

	CameraData cameraNodeData;
	glm::mat4 view;
	ArrayBuffer cameraNode;

	ArrayBuffer lightSources;
	vk::DescriptorSetLayout lightSourceLayout;
	vk::DescriptorSetLayout lightDescLayout;
	vk::PipelineLayout lightLayout;
	vk::Pipeline lightPipeline;
	vk::DescriptorSet lightMapSet;
	vk::DescriptorPool lightPool;

	vk::DescriptorSetLayout taaDescLayout;
	vk::PipelineLayout taaLayout;
	vk::Pipeline taaPipeline;
	std::vector<vk::DescriptorSet> taaSets;
	vk::DescriptorPool taaPool;

	vk::ImageView albedoView;
	vk::ImageView normalView;
	vk::ImageView normalView2;
	vk::Image resolve1Image;
	vk::Image resolve2Image;
	vk::ImageView resolveView;
	vk::ImageView resolveView2;

	vk::ImageView depthView;

	vk::Image depthImage;
	vk::Image normalImage1;
	vk::Image normalImage2;

	Trackball trackball;
	Settings settings;
	std::vector<LightSource> lights;

	glm::mat4 previousViewMatrix;

	Rng rng = rngInit(1, 2, 3, 4);

	ShadowMapContext shadowMapContext;

	std::vector<LightList> lightLists;
	ArrayBuffer lightListUpload;
	vk::DescriptorSetLayout lightListLayout;

	std::size_t numberOfLights = 3;
	std::size_t numberOfLightsWithShadowMaps = 3;
};

inline std::vector<uint8_t> loadFile(const std::string& value) {
	std::ifstream stream(value, std::ios::ate | std::ios::binary);
	if (!stream) {
		std::cout << "Could not load " << value << std::endl;
		return {};
	}
	const auto sizeOfFile = stream.tellg();
	stream.seekg(0, std::ios::beg);

	std::vector<uint8_t> values(sizeOfFile);
	stream.read((char*)values.data(), sizeOfFile);
	return values;
}

inline vk::PipelineShaderStageCreateInfo loadShaderStages(MainContext& context, const std::string& pPath) {
	const std::string path = (fs::path("..") / "shader" / pPath).string();
	vk::ShaderStageFlagBits shaderStage = vk::ShaderStageFlagBits::eVertex;
	if (path.ends_with(".frag.spv")) {
		shaderStage = vk::ShaderStageFlagBits::eFragment;
	}
	else if (path.ends_with(".rchit.spv")) {
		shaderStage = vk::ShaderStageFlagBits::eClosestHitKHR;
	}
	else if (path.ends_with(".rmiss.spv")) {
		shaderStage = vk::ShaderStageFlagBits::eMissKHR;
	}
	else if (path.ends_with(".rgen.spv")) {
		shaderStage = vk::ShaderStageFlagBits::eRaygenKHR;
	}
	const auto file = loadFile(path);
	if (file.empty()) {
		std::cout << "Could not load shader " << path << std::endl;
		return {};
	}
	vk::ShaderModuleCreateInfo shaderModuleInfo({}, file.size(), (uint32_t*)file.data());
	const auto sModule = context.device.createShaderModule(shaderModuleInfo);
	context.shaderModules.push_back(sModule);
	vk::PipelineShaderStageCreateInfo stageCreateInfo({}, shaderStage, sModule, "main");
	return stageCreateInfo;
}
