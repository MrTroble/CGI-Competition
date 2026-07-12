#pragma once

#include <vulkan/vulkan.hpp>
#include <filesystem>
#include "MainContext.hpp"
#include <tiny_gltf.h>
#include "glm/glm.hpp"

struct OutputModel {
	std::vector<vk::Buffer> buffers;
	std::vector<vk::DeviceMemory> memorys;
	std::vector<vk::Image> images;
	std::vector<vk::ImageView> imageViews;
	vk::CommandPool pool;
	std::vector<vk::CommandBuffer> cmd;
	std::vector<vk::Pipeline> pipes;
	std::vector<vk::Sampler> sampler;
	vk::DescriptorPool imageDescPool;
	std::vector<vk::DescriptorSet> imageSets;

	ArrayBuffer modelBuffer;
	ArrayBuffer rayTraceBuff;

	tinygltf::Model backupModel;

	RaytracingExtHolder rayHolder;
};

struct ModelConfig {
	vk::PolygonMode fillMode = vk::PolygonMode::eFill;
	glm::vec3 position;
	float scale = 1.0;
};

inline std::pair<size_t, size_t> getFromInput(tinygltf::Model& model, tinygltf::Primitive& value, const std::string& name) {
	const auto iter = value.attributes.find(name);
	if (iter == std::end(value.attributes))
		throw std::runtime_error("Could not find attribute!");
	const auto& indexAccess = model.accessors[iter->second];
	return std::make_pair(indexAccess.bufferView, indexAccess.byteOffset);
}

class GLTFLoader {

public:
	bool load(MainContext& context, const fs::path& path, OutputModel& out, const ModelConfig config);
};