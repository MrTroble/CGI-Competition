#pragma once

#include "glm/glm.hpp"
#include "Utilities.hpp"
#include "MainContext.hpp"
#include "rng.hpp"
#include <numeric>


void allocateShadowMaps(MainContext& context) {
	context.shadowMapContext.shadowMaps.reserve(NUMBER_OF_BIG_LIGHTS);
	context.shadowMapContext.shadowMapViews.reserve(NUMBER_OF_BIG_LIGHTS);

	vk::MemoryAllocateInfo memoryAlloc(0u, context.deviceLocalOnly);
	vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);
	for (std::size_t i = 0; i < NUMBER_OF_BIG_LIGHTS; i++) {
		vk::ImageCreateInfo shadowCreateInfo(
			{},
			vk::ImageType::e2D,
			vk::Format::eD16Unorm,
			vk::Extent3D{ context.currentExtent.width,  context.currentExtent.height, 1u },
			1u,
			1u,
			vk::SampleCountFlagBits::e1,
			vk::ImageTiling::eOptimal,
			vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst,
			vk::SharingMode::eExclusive,
			{}
		);
		const auto shadowImage = context.device.createImage(shadowCreateInfo);
		const auto requirements = context.device.getImageMemoryRequirements(shadowImage);
		memoryAlloc.allocationSize = requirements.size;
		const auto shadowImageMemory = context.device.allocateMemory(memoryAlloc);
		context.device.bindImageMemory(shadowImage, shadowImageMemory, 0);
		vk::ImageViewCreateInfo shadowViewCreate({}, shadowImage, vk::ImageViewType::e2D, vk::Format::eD16Unorm, {}, range);
		const auto shadowImageView = context.device.createImageView(shadowViewCreate);

		context.shadowMapContext.shadowMaps.push_back(shadowImage);
		context.shadowMapContext.shadowMapViews.push_back(shadowImageView);
	}
}

void createLightSources(MainContext& context) {
    std::vector<LightSource> lights(context.numberOfLights);

    const auto width = 50.0f;

    for (std::size_t i = 0; i < context.numberOfLights; i++) {
        glm::vec3 position = rngRndVec3(context.rng);
        position.x *= 2 * width;
        position.x -= width;
        position.z = position.z * 2.0f * width - width;
        position.y *= 20.0f;

		glm::vec3 direction = normalize(rngRndVec3(context.rng));
		float brightness = rngRndFloat(context.rng);
		glm::vec3 color = rngRndVec3(context.rng);
		if (i == 0) {
			position = glm::vec3(-0.9, 3.0, 0.0);
			brightness = 1.0;
			direction = glm::vec3(1.0, 0.0, 0.0);
		}
		if(i == 1) {
		    position = glm::vec3(7.0, 8.0, -10.0);
			brightness = 0.3;
			direction = glm::vec3(2.0, -2.0, 2.0);
			color = glm::vec3(1.0, 0.0, 1.0);
		}
		if(i == 2) {
            position = glm::vec3(7, 4, 6.0);
            brightness = 1.0;
            direction = glm::vec3(-1.0, -2.0, 0.0);
            color = glm::vec3(0.0, 0.0, 1.0);
		}

		LightSource light = LightSource{
			.position = position,
			.direction = direction,
			.color = color,
			.brightness = brightness
		};
		lights[i] = light;
	}

	// Allocate buffer
	const uint32_t byte_size = (uint32_t)(sizeof(LightSource) * NUMBER_OF_LIGHTS);
	const auto buffer = context.device.createBuffer(vk::BufferCreateInfo(
		{},
		byte_size + 8,
		vk::BufferUsageFlagBits::eUniformBuffer,
		vk::SharingMode::eExclusive
	));
	const auto bufferRequ = context.device.getBufferMemoryRequirements(buffer);

	vk::MemoryAllocateInfo memoryAlloc(bufferRequ.size, context.hostVisibleOnly != UINT32_MAX ? context.hostVisibleOnly : context.hostVisibleDeviceLocalMemory);
	const auto memory = context.device.allocateMemory(memoryAlloc);
	void* outputPtr = context.device.mapMemory(memory, 0, VK_WHOLE_SIZE);
	std::copy(lights.begin(), lights.end(), (LightSource*)outputPtr);
	*(uint32_t*)((uint8_t*)outputPtr + byte_size) = lights.size();
	*(uint32_t*)((uint8_t*)outputPtr + byte_size + 4) = context.numberOfLightsWithShadowMaps;
	context.device.unmapMemory(memory);
	context.device.bindBufferMemory(buffer, memory, 0);

	// Allocate buffer for shadow map matrices
	const uint32_t shadowMapMatrixSize = (uint32_t)(sizeof(glm::mat4) * NUMBER_OF_BIG_LIGHTS);
	const auto shadowMapMatrixBuffer = context.device.createBuffer(vk::BufferCreateInfo(
		{},
		shadowMapMatrixSize,
		vk::BufferUsageFlagBits::eUniformBuffer,
		vk::SharingMode::eExclusive
	));
	vk::MemoryAllocateInfo shadowMatrixMemoryAlloc(
		context.device.getBufferMemoryRequirements(shadowMapMatrixBuffer).size,
		context.hostVisibleOnly != UINT32_MAX ? context.hostVisibleOnly : context.hostVisibleDeviceLocalMemory
	);
	const auto shadowMapMatrixMemory = context.device.allocateMemory(shadowMatrixMemoryAlloc);
	context.device.bindBufferMemory(shadowMapMatrixBuffer, shadowMapMatrixMemory, 0);

	allocateShadowMaps(context);

	// Allocate pool
	const std::array poolSizes = { vk::DescriptorPoolSize(vk::DescriptorType::eUniformBuffer, 2), vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, NUMBER_OF_BIG_LIGHTS) };
	const auto poolCreateInfo = vk::DescriptorPoolCreateInfo({}, 1, poolSizes);
	auto pool = context.device.createDescriptorPool(poolCreateInfo);

    // Create descriptor sets
    std::vector<vk::DescriptorSetLayout> descriptorSetLayouts(1, context.lightSourceLayout);

    const auto descriptorAllocateInfo = vk::DescriptorSetAllocateInfo(pool, descriptorSetLayouts);
    const auto descriptors = context.device.allocateDescriptorSets(descriptorAllocateInfo);

	std::array<vk::DescriptorImageInfo, NUMBER_OF_BIG_LIGHTS> shadowMapImageInfos;
	for (std::size_t i = 0; i < NUMBER_OF_BIG_LIGHTS; i++) {
		const auto sampler = context.device.createSampler(vk::SamplerCreateInfo(
				{},
				vk::Filter::eNearest,
				vk::Filter::eNearest,
				vk::SamplerMipmapMode::eNearest,
				vk::SamplerAddressMode::eRepeat,
				vk::SamplerAddressMode::eRepeat,
				vk::SamplerAddressMode::eRepeat,
				{},
				{},
				{},
				true,
				vk::CompareOp::eLess
			));
		const auto imageInfo = vk::DescriptorImageInfo(sampler, context.shadowMapContext.shadowMapViews[i], vk::ImageLayout::eDepthStencilReadOnlyOptimal);
		shadowMapImageInfos[i] = imageInfo;
	}

	const auto bufferinfo = vk::DescriptorBufferInfo(buffer, 0, byte_size + 8);
	const auto shadowMatrixBufferInfo = vk::DescriptorBufferInfo(shadowMapMatrixBuffer, 0, shadowMapMatrixSize);
	const auto lightSourceListWrite = vk::WriteDescriptorSet(descriptors[0], 0, 0, vk::DescriptorType::eUniformBuffer, {}, bufferinfo);
	const auto shadowMapWrite = vk::WriteDescriptorSet(descriptors[0], 1, 0, vk::DescriptorType::eCombinedImageSampler, shadowMapImageInfos);
	const auto shadowMapMatrixWrite = vk::WriteDescriptorSet(descriptors[0], 2, 0, vk::DescriptorType::eUniformBuffer, {}, shadowMatrixBufferInfo);
	const auto writeDescriptorSets = { lightSourceListWrite, shadowMapWrite, shadowMapMatrixWrite };
	context.device.updateDescriptorSets(writeDescriptorSets, {});

    context.lightSources = ArrayBuffer{
        pool,
        descriptors,
        buffer
    };

	context.shadowMapContext.shadowMapMatrices = shadowMapMatrixMemory;

    static constexpr uint32_t LIGHT_LIST_AMOUNT = MAX_TILLING * MAX_TILLING * MAX_TILLING;
    LightList lightList;
    lightList.amount = lights.size();
    std::iota(std::begin(lightList.ids), std::begin(lightList.ids) + lightList.amount, 0u);
    context.lightLists.resize(LIGHT_LIST_AMOUNT, lightList);
    context.lightListUpload = createArrayBuffer<LightList>(context, context.lightLists, context.lightListLayout);
    context.lights = std::move(lights);
}

struct AABB {
    glm::vec3 min;
    glm::vec3 max;
};

bool overlapp(const AABB& aabb1, const AABB& aabb2) {
	return (aabb1.min.x <= aabb2.max.x && aabb1.max.x >= aabb2.min.x) &&
		   (aabb1.min.y <= aabb2.max.y && aabb1.max.y >= aabb2.min.y) &&
		   (aabb1.min.z <= aabb2.max.z && aabb1.max.z >= aabb2.min.z);
}

AABB clipAABB(const AABB& aabb) {
    return { glm::min(aabb.min, aabb.max), glm::max(aabb.min, aabb.max)};
}

inline uint32_t indexFrom(const glm::uvec3& tile, const RenderTillingInfo& info) {
    return tile.x + tile.y * info.xAmount + tile.z * info.xAmount * info.yAmount;
}

void rebuildLightLists(MainContext& context) {

    glm::vec3 leftOffset(10, 10, 10);
    glm::vec3 rightOffset = leftOffset * -1.0f;
    std::vector<AABB> screenSpaceAABBs(context.lights.size());
    std::transform(context.lights.begin(), context.lights.end(), screenSpaceAABBs.begin(),
        [&](LightSource& source) {
            return clipAABB(AABB{ source.position + leftOffset * source.brightness,
						 source.position + rightOffset * source.brightness });
        });
    const auto& tile = context.settings.baseTillingInfo;
	const auto vp = glm::inverse(context.cameraNodeData.mvp);
	glm::vec3 step = glm::vec3(2.0f, 2.0f, 1.0f) / glm::vec3(tile.xAmount, tile.yAmount, tile.zAmount);
	std::vector<AABB> aabbsOf(tile.xAmount * tile.yAmount * tile.zAmount);
	auto zStep = vp * glm::vec4(0.0f, 0.0f, step.z, 1.0f);
	float zSt = zStep.z / zStep.w;
	for (size_t x = 0; x < tile.xAmount; x++)
	{
		for (size_t y = 0; y < tile.yAmount; y++)
		{
			for (size_t z = 0; z < tile.zAmount; z++)
			{
				glm::vec3 min(std::numeric_limits<float>::max());
				glm::vec3 max(-std::numeric_limits<float>::max());
				for (size_t i = 0; i < 8; i++)
				{
					glm::uvec3 index(x + (i & 1 ? 1:0), y + (i & 2 ? 1 : 0), z + (i & 4 ? 1 : 0));
					auto position = step * glm::vec3(index) + glm::vec3(-1.0f, -1.0f, 0.0f);
					auto wPos = vp * glm::vec4(position, 1.0f);
					wPos /= wPos.w;
					min = glm::min(min, glm::vec3(wPos));
					max = glm::max(max, glm::vec3(wPos));
				}
				aabbsOf[indexFrom(glm::uvec3(x, y, z), tile)] = AABB{ min, max };
			}
		}
	}

    uint32_t currentLight = 0;
    for(auto& ll : context.lightLists) ll.amount = 0;
    for (const auto& aabb : screenSpaceAABBs) {
		for (size_t x = 0; x < tile.xAmount; x++)
		{
			for (size_t y = 0; y < tile.yAmount; y++)
			{
				for (size_t z = 0; z < tile.zAmount; z++)
				{
					const auto index = indexFrom(glm::uvec3(x, y, z), tile);
					auto& aabbFrustum = aabbsOf[index];
					if (overlapp(aabb, aabbFrustum)) {
						auto& ll = context.lightLists[index];
						ll.ids[ll.amount++] = currentLight;
					}
					else {
						continue;
					}
				}
			}
		}
		currentLight++;
    }
    
    auto startPtr = (uint8_t*)context.lightLists.data();
    updateMemory(context, context.lightListUpload.memory, std::span{ startPtr, context.lightLists.size() * sizeof(LightList)});
}
