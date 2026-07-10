#pragma once

#include <span>
#include <vector>
#include "MainContext.hpp"

inline void updateMemory(MainContext& context, vk::DeviceMemory memory, std::span<const uint8_t> values) {
	void* outputPtr = context.device.mapMemory(memory, 0, VK_WHOLE_SIZE);
	std::copy(values.begin(), values.end(), (uint8_t*)outputPtr);
	context.device.unmapMemory(memory);
}

inline vk::DeviceMemory loadToMemory(MainContext& context, std::span<const uint8_t> values, const vk::MemoryRequirements& memoryReq) {
	vk::MemoryAllocateInfo memoryAlloc(memoryReq.size, context.hostVisibleOnly != UINT32_MAX ? context.hostVisibleOnly:context.hostVisibleDeviceLocalMemory);
	vk::MemoryAllocateFlagsInfo memAllocInfo(vk::MemoryAllocateFlagBits::eDeviceAddress);
	memoryAlloc.pNext = memAllocInfo;
	const auto memory = context.device.allocateMemory(memoryAlloc);
	updateMemory(context, memory, values);
	return memory;
}

template<typename T>
inline ArrayBuffer createArrayBuffer(MainContext& context, std::span<T> data, vk::DescriptorSetLayout layout) {
	// Allocate buffer
	const auto buffer = context.device.createBuffer(vk::BufferCreateInfo(
		{},
		(uint32_t)data.size_bytes(),
		vk::BufferUsageFlagBits::eUniformBuffer,
		vk::SharingMode::eExclusive
	));
	const auto bufferRequ = context.device.getBufferMemoryRequirements(buffer);
	const auto memory = loadToMemory(context, { (uint8_t*)data.data(), data.size_bytes()}, bufferRequ);
	context.device.bindBufferMemory(buffer, memory, 0);

	// Allocate pool
	const std::array poolSizes = { vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, data.size()) };
	const auto poolCreateInfo = vk::DescriptorPoolCreateInfo({}, (uint32_t)data.size(), poolSizes);
	auto pool = context.device.createDescriptorPool(poolCreateInfo);

	// Create descriptor sets
	std::vector<vk::DescriptorSetLayout> descriptorSetLayouts(data.size(), layout);

	const auto descriptorAllocateInfo = vk::DescriptorSetAllocateInfo(pool, descriptorSetLayouts);
	const auto descriptors = context.device.allocateDescriptorSets(descriptorAllocateInfo);

	// Bind descriptors to buffer
	std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
	std::vector<vk::DescriptorBufferInfo> bufferInfos(data.size());
	writeDescriptorSets.reserve(data.size());
	for (std::size_t i = 0; i < data.size(); i++) {
		const std::size_t offset = sizeof(T) * i;
		const auto bufferInfo = vk::DescriptorBufferInfo(buffer, offset, sizeof(T));
		bufferInfos[i] = bufferInfo;
		const auto writeDescriptorSet = vk::WriteDescriptorSet(descriptors[i], 0, 0, 1, vk::DescriptorType::eUniformBuffer, nullptr, &bufferInfos[i]);
		writeDescriptorSets.push_back(writeDescriptorSet);
	}
	context.device.updateDescriptorSets(writeDescriptorSets, {});


	return ArrayBuffer {
		pool,
		descriptors,
		buffer,
		memory
	};
}
