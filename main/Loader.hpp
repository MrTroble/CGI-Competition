#pragma once

#include <vulkan/vulkan.hpp>
#include <filesystem>
#include "MainContext.hpp"
#include <tiny_gltf.h>
#include "glm/glm.hpp"
#include "Shared.hpp"

struct UVCopyInfo {
	DataBuffer buffer;
	size_t offset;
	size_t size;
	bool markedForDeletion = false;
};

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

	std::vector<MaterialData> rtxMaterials;
	std::vector<GeometryData> rtxGeodata;
	std::vector<UVCopyInfo> rtxCopyData;
	std::vector<UVCopyInfo> rtxIndexCopyData;
	std::vector<UVCopyInfo> rtxNormalCopyData;
	std::vector<UVCopyInfo> rtxVertexCopyData;

	size_t completeAmountToCopy = 0;
	size_t completeAmountToIndexCopy = 0;
	size_t completeAmountToNormalCopy = 0;
	size_t completeAmountToVertexCopy = 0;
};

struct ModelConfig {
	vk::PolygonMode fillMode = vk::PolygonMode::eFill;
	glm::vec3 position{};
	float scale = 1.0;
	size_t shaderBindingOffset = 0;
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
	bool load(MainContext& context, const fs::path& path, OutputModel& out, const ModelConfig config, uint32_t blasIndex = 0);
};