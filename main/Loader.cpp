#include "Loader.hpp"
#include "Utilities.hpp"

#include <iostream>
#include <span>
#include <vector>
#include <glm/glm.hpp>
#include "glm/gtc/matrix_transform.hpp"


inline vk::Format fromBitsPerChannel(int bits) {
	if (bits == 8)
		return vk::Format::eR8G8B8A8Unorm;
	if (bits == 16)
		return vk::Format::eR16G16B16A16Unorm;
	if (bits == 32)
		return vk::Format::eR32G32B32A32Sfloat;
	throw std::runtime_error("Format could not be found!");
}

inline vk::Pipeline createPipeline(MainContext& context, const bool doubeleSided, const std::string& usedSystem, const vk::PolygonMode fillMode = vk::PolygonMode::eFill) {

	std::string shaderPermute = (usedSystem + (usedSystem.empty() ? "" : "_")) + "geometry.frag.spv";
	std::array shaderStagesLoaded = {
		loadShaderStages(context, "default.vert.spv"),
		loadShaderStages(context, shaderPermute)
	};

	vk::VertexInputBindingDescription inputBindingPos(0, sizeof(glm::vec3));
	vk::VertexInputBindingDescription inputBindingNormal(1, sizeof(glm::vec3));
	vk::VertexInputBindingDescription inputBindingTex1(2, sizeof(glm::vec2));

	std::array vertexBindings = { inputBindingPos, inputBindingNormal, inputBindingTex1 };

	vk::VertexInputAttributeDescription descPos(0, 0, vk::Format::eR32G32B32Sfloat, 0);
	vk::VertexInputAttributeDescription descNormal(1, 1, vk::Format::eR32G32B32Sfloat, 0);
	vk::VertexInputAttributeDescription descTex1(2, 2, vk::Format::eR32G32Sfloat, 0);

	std::array vertexAttribution = { descPos, descNormal, descTex1 };

	vk::PipelineVertexInputStateCreateInfo vertexInputState({}, vertexBindings, vertexAttribution);

	vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleList, false);

	vk::Viewport viewportIn(0, 0, context.currentExtent.width, context.currentExtent.height, 0, 1.0f);
	vk::Rect2D rect2d({}, context.currentExtent);
	vk::PipelineViewportStateCreateInfo viewport({}, viewportIn, rect2d);

	vk::PipelineRasterizationStateCreateInfo rasterizationCreateInfo({}, false, false, fillMode,
		doubeleSided ? vk::CullModeFlagBits::eNone : vk::CullModeFlagBits::eBack);
	rasterizationCreateInfo.lineWidth = 1.0f;

	vk::PipelineMultisampleStateCreateInfo multisampleCreate(vk::PipelineMultisampleStateCreateFlags{});

	vk::PipelineDepthStencilStateCreateInfo depthStencil({}, true, true, vk::CompareOp::eLess);

	std::array<vk::PipelineColorBlendAttachmentState, 2> attachmentCount;
	// TODO configure
	attachmentCount[0].colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;
	attachmentCount[1].colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;

	vk::PipelineColorBlendStateCreateInfo pipelineColorBlend({}, false, {}, attachmentCount);

	vk::GraphicsPipelineCreateInfo pipeCreateInfo({}, shaderStagesLoaded, &vertexInputState,
		&inputAssembly, {}, &viewport, &rasterizationCreateInfo, &multisampleCreate, &depthStencil, &pipelineColorBlend, {},
		context.pipelineLayout, context.renderPass, 0);
	const auto result = context.device.createGraphicsPipeline({}, pipeCreateInfo);
	// TODO Check
	return result.value;
}

inline glm::mat4 matrixFromNode(tinygltf::Node& node) {
	if (node.matrix.empty()) {
		return glm::mat4(1.0f);
	}
	else {
		glm::mat4 matrix;
		for (size_t i = 0; i < node.matrix.size(); i++)
		{
			*((float*)&matrix + i) = (float)node.matrix[i];
		}
		return matrix;
	}
}


inline void imageDescriptors(MainContext& context, OutputModel& out) {
	vk::DescriptorPoolSize mainPoolSize(vk::DescriptorType::eSampledImage, out.images.size());
	vk::DescriptorPoolCreateInfo poolCreateInfo({}, out.images.size());
	const auto pool = context.device.createDescriptorPool(poolCreateInfo);
	std::vector layoutsToUse(out.images.size(), context.textureLayout);
	const vk::DescriptorSetAllocateInfo allocateInfo(pool, layoutsToUse);
	const auto descriptors = context.device.allocateDescriptorSets(allocateInfo);

	std::vector<vk::WriteDescriptorSet> writeDescriptorSets;
	writeDescriptorSets.reserve(out.images.size());
	for (std::size_t i = 0; i < out.images.size(); i++) {
		const auto imageView = context.device.createImageView(vk::ImageViewCreateInfo(
			{},
			out.images[i],
			vk::ImageViewType::e2D,
			vk::Format::eR8G8B8A8Unorm,
			vk::ComponentMapping(), // Identity mapping on every component
			vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor)
		));
		const auto imageInfo = vk::DescriptorImageInfo({}, imageView, vk::ImageLayout::eColorAttachmentOptimal);
		const auto writeDescriptorSet = vk::WriteDescriptorSet(descriptors[i], 0, i, 1, vk::DescriptorType::eSampledImage, &imageInfo, nullptr);
		writeDescriptorSets.push_back(writeDescriptorSet);
	}
	context.device.updateDescriptorSets(writeDescriptorSets, {});
}

bool GLTFLoader::load(MainContext& context, const fs::path& pPath, OutputModel& out, const ModelConfig config)
{
	const fs::path path = fs::absolute(fs::path("..") / pPath);
	using namespace tinygltf;
	if (!fs::exists(path)) {
		std::cout << "Path does not exist: " << path << std::endl;
		return false;
	}

	Model& model = out.backupModel;
	TinyGLTF loader;
	std::string err;
	std::string warn;

	bool ret = false;
	if (!path.string().ends_with(".glb")) {
		ret = loader.LoadASCIIFromFile(&model, &err, &warn, path.string());
	}
	else {
		ret = loader.LoadBinaryFromFile(&model, &err, &warn, path.string());
	}

	if (!warn.empty()) {
		printf("Warn: %s\n", warn.c_str());
	}

	if (!err.empty()) {
		printf("Err: %s\n", err.c_str());
	}

	if (!ret) {
		printf("Failed to parse glTF: %s\n", path.string().c_str());
		return false;
	}

	constexpr vk::BufferUsageFlags usageFlags = vk::BufferUsageFlagBits::eIndexBuffer | vk::BufferUsageFlagBits::eVertexBuffer
		| vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
	const std::array queueIndexes = { context.queueIndex };

	for (auto& buffer : model.bufferViews)
	{
		if (buffer.target == 0) {
			out.buffers.push_back({});
			out.memorys.push_back({});
			continue;
		}
		const auto& actualData = model.buffers[buffer.buffer];
		vk::BufferCreateInfo createInfo({}, buffer.byteLength, usageFlags, vk::SharingMode::eExclusive, queueIndexes);
		const auto vbuffer = context.device.createBuffer(createInfo);
		const auto bufferRequ = context.device.getBufferMemoryRequirements(vbuffer);

		const auto startOff = actualData.data.begin() + buffer.byteOffset;
		const auto memory = loadToMemory(context, { startOff, startOff + buffer.byteLength }, bufferRequ);
		context.device.bindBufferMemory(vbuffer, memory, 0);

		out.buffers.push_back(vbuffer);
		out.memorys.push_back(memory);
	}

	std::cout << "Finished buffer loads" << std::endl;

	const auto usageFlagsImage = vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst;
	vk::MemoryAllocateInfo memoryAlloc(0u, context.deviceLocalOnly);

	vk::CommandPoolCreateInfo poolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, context.queueIndex);
	const auto pool = context.device.createCommandPool(poolCreateInfo);
	vk::CommandBufferAllocateInfo allocateInfo(pool, vk::CommandBufferLevel::ePrimary, 1);
	const auto cmd = context.device.allocateCommandBuffers(allocateInfo).back();
	vk::CommandBufferBeginInfo cmdBeginInfo(vk::CommandBufferUsageFlags{});
	cmd.begin(cmdBeginInfo);

	std::vector<vk::Buffer> localBuffer;
	std::vector<vk::DeviceMemory> localMemory;

	vk::ImageSubresourceLayers imageLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);

	vk::BufferImageCopy bufferImageCopy(0, 0u, 0u, imageLayers, { 0,0,0 }, { 1u, 1u, 1u });
	std::vector<vk::BufferImageCopy> bufferImageCopys(model.images.size());

	uint32_t currentCount = 0;
	constexpr vk::BufferUsageFlags usageFlagsTempBuffer = vk::BufferUsageFlagBits::eTransferSrc;

	vk::ImageSubresourceRange subRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

	std::vector<vk::ImageMemoryBarrier> barriers(2 * model.images.size());

	vk::ImageMemoryBarrier barrier1(vk::AccessFlagBits::eNone, vk::AccessFlagBits::eTransferWrite,
		vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal, context.queueIndex, context.queueIndex,
		{}, subRange);

	vk::ImageMemoryBarrier barrier2(vk::AccessFlagBits::eTransferWrite, vk::AccessFlagBits::eShaderRead,
		vk::ImageLayout::eTransferDstOptimal, vk::ImageLayout::eShaderReadOnlyOptimal, context.queueIndex, context.queueIndex,
		{}, subRange);

	std::vector<vk::Format> formats;
	for (auto& image : model.images) {
		if (image.component != 4) {
			std::cout << "Component amount should be 4" << std::endl;
			return false;
		}

		vk::Buffer vbuffer;
		vk::BufferCreateInfo bInfo({}, image.image.size(), usageFlagsTempBuffer, vk::SharingMode::eExclusive, queueIndexes);
		vbuffer = context.device.createBuffer(bInfo);
		const auto bufferRequ = context.device.getBufferMemoryRequirements(vbuffer);

		const auto hostMem = loadToMemory(context, image.image, bufferRequ);
		context.device.bindBufferMemory(vbuffer, hostMem, 0);

		localBuffer.push_back(vbuffer);
		localMemory.push_back(hostMem);

		const auto format = fromBitsPerChannel(image.bits);
		formats.push_back(format);
		vk::ImageCreateInfo createInfo({}, vk::ImageType::e2D, format, vk::Extent3D{ (uint32_t)image.width, (uint32_t)image.height, 1u }
		, 1u, 1u, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, usageFlagsImage, vk::SharingMode::eExclusive, queueIndexes);
		const auto vImage = context.device.createImage(createInfo);
		const auto imgRequ = context.device.getImageMemoryRequirements(vImage);
		memoryAlloc.allocationSize = imgRequ.size;

		const auto memory = context.device.allocateMemory(memoryAlloc);

		context.device.bindImageMemory(vImage, memory, 0);

		bufferImageCopy.bufferImageHeight = image.height;
		bufferImageCopy.bufferRowLength = image.width;
		bufferImageCopy.imageExtent.width = image.width;
		bufferImageCopy.imageExtent.height = image.height;
		bufferImageCopys[currentCount] = bufferImageCopy;

		barrier1.image = vImage;
		barrier2.image = vImage;
		barriers[currentCount] = barrier1;
		auto& b2 = barriers[model.images.size() + currentCount];
		b2 = barrier2;
		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics, vk::PipelineStageFlagBits::eTransfer, vk::DependencyFlagBits::eByRegion,
			{}, {}, barriers[currentCount]);

		cmd.copyBufferToImage(vbuffer, vImage, vk::ImageLayout::eTransferDstOptimal, bufferImageCopys[currentCount]);

		cmd.pipelineBarrier(vk::PipelineStageFlagBits::eTransfer, vk::PipelineStageFlagBits::eAllGraphics, vk::DependencyFlagBits::eByRegion,
			{}, {}, b2);

		out.images.push_back(vImage);
		out.memorys.push_back(memory);
		currentCount++;
	}

	cmd.end();
	std::cout << "Finished command buffer loading" << std::endl;

	vk::SubmitInfo submitInfo({}, {}, cmd);
	context.queue.submit(submitInfo, context.fence);

	size_t imageIndex = 0;
	for (auto image : out.images) {
		vk::ImageViewCreateInfo imageViewCreate({}, image, vk::ImageViewType::e2D, formats[imageIndex], {}, subRange);
		const auto view = context.device.createImageView(imageViewCreate);
		out.imageViews.push_back(view);
		imageIndex++;
	}

	std::cout << "Finished image primary" << std::endl;

	out.pool = context.device.createCommandPool(poolCreateInfo);
	vk::CommandBufferAllocateInfo allocateInfoSec(out.pool, vk::CommandBufferLevel::eSecondary, context.framebuffer.size());
	out.cmd = context.device.allocateCommandBuffers(allocateInfoSec);
	vk::CommandBufferInheritanceInfo inharitance(context.renderPass, 0);
	cmdBeginInfo.pInheritanceInfo = &inharitance;
	cmdBeginInfo.setFlags(vk::CommandBufferUsageFlagBits::eRenderPassContinue);

	const auto transformMatrix = glm::scale(glm::translate(glm::mat4(1.0f), config.position), glm::vec3(config.scale));
	std::vector<PerNodeData> nodeMatrix(model.nodes.size(), { transformMatrix });
	std::vector<vk::AccelerationStructureGeometryKHR> geometryData;
	std::vector<vk::AccelerationStructureGeometryKHR> instanceBasedData;
	std::vector<uint32_t> maxPrimitive;
	std::vector<vk::AccelerationStructureBuildRangeInfoKHR> structureBuildRange;
	size_t perNodeId = 0;
	for (auto& node : model.nodes) {
		PerNodeData& nodeData = nodeMatrix[perNodeId++];
		nodeData.mvp *= matrixFromNode(node);
		for (auto childID : node.children) {
			nodeMatrix[childID].mvp = nodeData.mvp;
		}
		nodeData.previousViewMatrix = glm::transpose(glm::inverse(nodeData.mvp));
		if (node.mesh >= 0) {
			auto& mesh = model.meshes[node.mesh];
			for (auto& primitive : mesh.primitives) {
				vk::AccelerationStructureGeometryKHR currentData;
				const auto& indexAccess = model.accessors[primitive.indices];
				currentData.geometry.triangles.indexType = (indexAccess.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? vk::IndexType::eUint16 : vk::IndexType::eUint32);
				const auto addressIndex = context.device.getBufferAddress({ out.buffers[indexAccess.bufferView] }, context.dynamicLoader);
				currentData.geometry.triangles.indexData.deviceAddress = addressIndex;
				currentData.geometry.triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
				const auto [posId, posOff] = getFromInput(model, primitive, "POSITION");
				const auto addressVertex = context.device.getBufferAddress({ out.buffers[posId] }, context.dynamicLoader);
				currentData.geometry.triangles.vertexData = addressVertex + posOff;
				currentData.geometry.triangles.vertexStride = sizeof(glm::vec3);
				currentData.geometry.triangles.maxVertex = indexAccess.count - 1;
				const uint32_t primitiveSize = (uint32_t)indexAccess.count / 3;
				geometryData.push_back(currentData);
				maxPrimitive.push_back(primitiveSize);
				vk::AccelerationStructureBuildRangeInfoKHR primitveRanges{ primitiveSize, 0, 0, 0 };
				structureBuildRange.push_back(primitveRanges);
			}
		}
	}
	out.modelBuffer = createArrayBuffer<PerNodeData>(context, nodeMatrix, context.modelLayout);

	RaytracingExtHolder extHolder;

	std::array<vk::AccelerationStructureBuildGeometryInfoKHR, 1> geometryInfo = {
		vk::AccelerationStructureBuildGeometryInfoKHR{vk::AccelerationStructureTypeKHR::eBottomLevel}
	   //,vk::AccelerationStructureBuildGeometryInfoKHR{vk::AccelerationStructureTypeKHR::eTopLevel} 
	};
	std::array<vk::AccelerationStructureBuildRangeInfoKHR*, 1> buildRanges;
	buildRanges[0] = structureBuildRange.data();
	geometryInfo[0].setGeometries(geometryData);
	size_t indexLocal = 0;
	for (const auto& buildGeo : geometryInfo) {
		geometryInfo[indexLocal].mode = vk::BuildAccelerationStructureModeKHR::eBuild;
		geometryInfo[indexLocal].flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
		const auto buildSizeInfo = context.device.getAccelerationStructureBuildSizesKHR(
			vk::AccelerationStructureBuildTypeKHR::eDevice, geometryInfo[indexLocal], maxPrimitive, context.dynamicLoader);
		vk::BufferCreateFlags localFlags = {};
		const auto bufferUsage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
		vk::BufferCreateInfo createInfo(localFlags, buildSizeInfo.accelerationStructureSize, bufferUsage, vk::SharingMode::eExclusive, queueIndexes);
		const auto vbuffer = context.device.createBuffer(createInfo);
		const auto bufferRequ = context.device.getBufferMemoryRequirements(vbuffer);

		vk::MemoryAllocateFlagsInfo blasFlagInfo{};
		blasFlagInfo.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;

		vk::MemoryAllocateInfo allocInfo(bufferRequ.size, context.deviceLocalOnly);
		allocInfo.pNext = &blasFlagInfo;
		const auto memory = context.device.allocateMemory(allocInfo);
		context.device.bindBufferMemory(vbuffer, memory, 0);

		vk::MemoryAllocateFlagsInfo flagInfo(vk::MemoryAllocateFlagBits::eDeviceAddress);
		allocInfo.pNext = &flagInfo;
		createInfo.usage = vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eStorageBuffer;
		createInfo.setSize(buildSizeInfo.buildScratchSize);
		const auto scratchbuffer = context.device.createBuffer(createInfo);
		const auto scratchbufferRequ = context.device.getBufferMemoryRequirements(vbuffer);

		allocInfo.allocationSize = scratchbufferRequ.size;
		const auto memoryScratch = context.device.allocateMemory(allocInfo);
		context.device.bindBufferMemory(scratchbuffer, memoryScratch, 0);
		extHolder.scratch[indexLocal] = scratchbuffer;
		extHolder.memory.push_back(memoryScratch);
		extHolder.memory.push_back(memory);

		extHolder.buffer[indexLocal] = vbuffer;

		vk::AccelerationStructureCreateInfoKHR accelerationCreateInfo(vk::AccelerationStructureCreateFlagBitsKHR::eDeviceAddressCaptureReplay,
			vbuffer, 0, buildSizeInfo.accelerationStructureSize, geometryInfo[indexLocal].type);
		const auto dataStructure = context.device.createAccelerationStructureKHR(accelerationCreateInfo, nullptr, context.dynamicLoader);
		geometryInfo[indexLocal].dstAccelerationStructure = dataStructure;
		geometryInfo[indexLocal].scratchData = context.device.getBufferAddress({ scratchbuffer }, context.dynamicLoader);
		extHolder.accelerationStructures[indexLocal] = dataStructure;
		indexLocal++;
	}

	context.device.waitForFences(context.fence, true, UINT64_MAX);
	context.device.resetFences(context.fence);
	cmd.reset();

	vk::CommandBufferBeginInfo beginInfoTwo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	cmd.begin(beginInfoTwo);

	cmd.buildAccelerationStructuresKHR(geometryInfo, buildRanges, context.dynamicLoader);

	cmd.end();
	context.queue.submit(submitInfo, context.fence);

	out.rayHolder = extHolder;

	vk::AccelerationStructureDeviceAddressInfoKHR blasAddressInfo{};
	blasAddressInfo.accelerationStructure = extHolder.accelerationStructures[0];
	vk::DeviceAddress blasAddress = context.device.getAccelerationStructureAddressKHR(blasAddressInfo, context.dynamicLoader);
	vk::AccelerationStructureInstanceKHR tlasInstance{};
	glm::mat4 transform = transformMatrix;
	memcpy(&tlasInstance.transform, &transform, sizeof(tlasInstance.transform));
	tlasInstance.instanceCustomIndex = 0;
	tlasInstance.mask = 0xFF;
	tlasInstance.instanceShaderBindingTableRecordOffset = 0;
	tlasInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
	tlasInstance.accelerationStructureReference = blasAddress;

	vk::BufferCreateInfo tlasInstanceBufferInfo{};
	tlasInstanceBufferInfo.size = sizeof(vk::AccelerationStructureInstanceKHR);
	tlasInstanceBufferInfo.usage = vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	vk::Buffer instanceBuffer = context.device.createBuffer(tlasInstanceBufferInfo);

	auto requirements = context.device.getBufferMemoryRequirements(instanceBuffer);
	vk::MemoryAllocateFlagsInfo tlasFlags{};
	tlasFlags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	vk::MemoryAllocateInfo alloc{};
	alloc.allocationSize = requirements.size;
	alloc.memoryTypeIndex = context.hostVisibleDeviceLocalMemory;
	alloc.pNext = &tlasFlags;
	vk::DeviceMemory instanceMemory = context.device.allocateMemory(alloc);

	context.device.bindBufferMemory(instanceBuffer, instanceMemory, 0);
	void* data = context.device.mapMemory(instanceMemory, 0, sizeof(tlasInstance));
	memcpy(data, &tlasInstance, sizeof(tlasInstance));
	context.device.unmapMemory(instanceMemory);
	vk::DeviceAddress instanceAddress = context.device.getBufferAddress({ instanceBuffer }, context.dynamicLoader);

	vk::AccelerationStructureGeometryInstancesDataKHR instancesData{};

	instancesData.arrayOfPointers = VK_FALSE;
	instancesData.data.deviceAddress = instanceAddress;

	vk::AccelerationStructureGeometryKHR tlasGeometry{};
	tlasGeometry.geometryType = vk::GeometryTypeKHR::eInstances;
	tlasGeometry.geometry.instances = instancesData;

	vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{};
	tlasBuildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
	tlasBuildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
	tlasBuildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
	tlasBuildInfo.geometryCount = 1;
	tlasBuildInfo.pGeometries = &tlasGeometry;
	tlasBuildInfo.dstAccelerationStructure = extHolder.accelerationStructures[1];
	uint32_t instanceCount = 1;

	auto tlasSize = context.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, tlasBuildInfo, instanceCount, context.dynamicLoader);

	vk::BufferCreateInfo tlasBufferInfo{};
	tlasBufferInfo.size = tlasSize.accelerationStructureSize;
	tlasBufferInfo.usage = vk::BufferUsageFlagBits::eAccelerationStructureStorageKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	tlasBufferInfo.sharingMode = vk::SharingMode::eExclusive;
	vk::Buffer tlasBuffer = context.device.createBuffer(tlasBufferInfo);
	auto tlasRequirements = context.device.getBufferMemoryRequirements(tlasBuffer);
	vk::MemoryAllocateFlagsInfo tlasMemoryFlags{}; tlasMemoryFlags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	vk::MemoryAllocateInfo tlasAlloc{};
	tlasAlloc.allocationSize = tlasRequirements.size;
	tlasAlloc.memoryTypeIndex = context.deviceLocalOnly;
	tlasAlloc.pNext = &tlasMemoryFlags;
	vk::DeviceMemory tlasMemory = context.device.allocateMemory(tlasAlloc);
	context.device.bindBufferMemory(tlasBuffer, tlasMemory, 0);

	vk::AccelerationStructureCreateInfoKHR tlasCreateInfo{};
	tlasCreateInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
	tlasCreateInfo.size = tlasSize.accelerationStructureSize;
	tlasCreateInfo.buffer = tlasBuffer;

	extHolder.accelerationStructures[1] = context.device.createAccelerationStructureKHR(tlasCreateInfo, nullptr, context.dynamicLoader);;

	vk::BufferCreateInfo scratchInfo{};
	scratchInfo.size = tlasSize.buildScratchSize;
	scratchInfo.usage = vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eShaderDeviceAddress;
	vk::Buffer tlasScratch = context.device.createBuffer(scratchInfo);
	auto scratchReq = context.device.getBufferMemoryRequirements(tlasScratch);
	vk::MemoryAllocateFlagsInfo scratchFlags{};
	scratchFlags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	vk::MemoryAllocateInfo scratchAlloc{};
	scratchAlloc.allocationSize = scratchReq.size;
	scratchAlloc.memoryTypeIndex = context.deviceLocalOnly;
	scratchAlloc.pNext = &scratchFlags;
	vk::DeviceMemory scratchMemory = context.device.allocateMemory(scratchAlloc);
	context.device.bindBufferMemory(tlasScratch, scratchMemory, 0);
	tlasBuildInfo.dstAccelerationStructure = extHolder.accelerationStructures[1];
	tlasBuildInfo.scratchData = context.device.getBufferAddress({ tlasScratch }, context.dynamicLoader);

	vk::AccelerationStructureBuildRangeInfoKHR tlasRange{};
	tlasRange.primitiveCount = 1;
	std::array ranges = { &tlasRange };


	context.device.waitForFences(context.fence, true, UINT64_MAX);
	context.device.resetFences(context.fence);
	cmd.reset();

	vk::CommandBufferBeginInfo beginInfoTlas(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	cmd.begin(beginInfoTlas);

	cmd.buildAccelerationStructuresKHR(tlasBuildInfo, ranges, context.dynamicLoader);

	cmd.end();
	context.queue.submit(submitInfo, context.fence);

	vk::DescriptorSetAllocateInfo allocInfo{};
	allocInfo.descriptorPool = context.raytraceExt.pool;
	allocInfo.descriptorSetCount = 1;
	allocInfo.setSetLayouts(context.raytraceExt.descLayout);

	vk::DescriptorSet descriptorSet = context.device.allocateDescriptorSets(allocInfo)[0];

	vk::WriteDescriptorSetAccelerationStructureKHR asInfo{};
	asInfo.accelerationStructureCount = 1;
	asInfo.pAccelerationStructures = &extHolder.accelerationStructures[1];

	vk::WriteDescriptorSet writeAS{};
	writeAS.dstBinding = 0;
	writeAS.descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
	writeAS.pNext = &asInfo;
	writeAS.dstSet = descriptorSet;
	writeAS.descriptorCount = 1;

	vk::ImageCreateInfo imageInfo{};
	imageInfo.imageType = vk::ImageType::e2D;
	imageInfo.format = vk::Format::eR8G8B8A8Unorm;
	imageInfo.extent = vk::Extent3D(context.currentExtent.width, context.currentExtent.height, 1);
	imageInfo.mipLevels = 1;
	imageInfo.arrayLayers = 1;
	imageInfo.samples = vk::SampleCountFlagBits::e1;
	imageInfo.tiling = vk::ImageTiling::eOptimal;
	imageInfo.usage = vk::ImageUsageFlagBits::eStorage | vk::ImageUsageFlagBits::eTransferSrc;

	vk::Image outputImage = context.device.createImage(imageInfo);
	vk::MemoryRequirements memRequirements = context.device.getImageMemoryRequirements(outputImage);
	vk::MemoryAllocateInfo allocInfoImage{ memRequirements.size , context.deviceLocalOnly };
	vk::DeviceMemory outputMemory = context.device.allocateMemory(allocInfoImage);
	context.device.bindImageMemory(outputImage, outputMemory, 0);

	vk::ImageViewCreateInfo viewInfo{};
	viewInfo.image = outputImage;
	viewInfo.viewType = vk::ImageViewType::e2D;
	viewInfo.format = vk::Format::eR8G8B8A8Unorm;

	viewInfo.subresourceRange.aspectMask =
		vk::ImageAspectFlagBits::eColor;
	viewInfo.subresourceRange.baseMipLevel = 0;
	viewInfo.subresourceRange.levelCount = 1;
	viewInfo.subresourceRange.baseArrayLayer = 0;
	viewInfo.subresourceRange.layerCount = 1;

	vk::ImageView outputImageView = context.device.createImageView(viewInfo);

	vk::DescriptorImageInfo desImageInfo{};
	desImageInfo.imageView = outputImageView;
	desImageInfo.imageLayout = vk::ImageLayout::eGeneral;

	vk::WriteDescriptorSet writeImage{};
	writeImage.dstBinding = 1;
	writeImage.descriptorType = vk::DescriptorType::eStorageImage;
	writeImage.pImageInfo = &desImageInfo;
	writeImage.dstSet = descriptorSet;
	writeImage.descriptorCount = 1;

	context.device.updateDescriptorSets({ writeAS, writeImage }, {});

	vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties{};
	vk::PhysicalDeviceProperties2 properties{};
	properties.pNext = &rayTracingProperties;

	context.usedPhyDevice.getProperties2(&properties);

	uint32_t handleSize = rayTracingProperties.shaderGroupHandleSize;
	//todo remove magic number
	uint32_t groupCount = 3;
	std::vector<uint8_t> shaderHandles(groupCount * handleSize);
	auto handels = context.device.getRayTracingShaderGroupHandlesKHR(context.raytraceExt.mainRayPipeline, 0, groupCount, shaderHandles.size(), shaderHandles.data(), context.dynamicLoader);
	vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst;

	uint32_t handleAlignment = rayTracingProperties.shaderGroupHandleAlignment;
	uint32_t baseAlignment = rayTracingProperties.shaderGroupBaseAlignment;
	auto alignUp = [](uint32_t value, uint32_t alignment)
		{
			return (value + alignment - 1) &
				~(alignment - 1);
		};
	uint32_t handleStride = alignUp(handleSize, handleAlignment);
	uint32_t raygenSize = handleStride;
	uint32_t missSize = handleStride;
	uint32_t hitSize = handleStride;
	uint32_t missOffset = alignUp(raygenSize, baseAlignment);
	uint32_t hitOffset = alignUp(missOffset + missSize, baseAlignment);
	uint32_t sbtSize = hitOffset + hitSize;

	vk::BufferCreateInfo sbtBufferInfo{};
	sbtBufferInfo.size = sbtSize;
	sbtBufferInfo.usage = usage;
	sbtBufferInfo.sharingMode = vk::SharingMode::eExclusive;
	vk::Buffer sbtBuffer = context.device.createBuffer(sbtBufferInfo);

	vk::MemoryRequirements memReq = context.device.getBufferMemoryRequirements(sbtBuffer);
	vk::MemoryAllocateFlagsInfo flags{};
	flags.flags = vk::MemoryAllocateFlagBits::eDeviceAddress;
	vk::MemoryAllocateInfo allocInfoSBT{};

	allocInfoSBT.allocationSize = memReq.size;
	allocInfoSBT.memoryTypeIndex = context.hostVisibleDeviceLocalMemory;
	allocInfoSBT.pNext = &flags;

	vk::DeviceMemory sbtMemory = context.device.allocateMemory(allocInfoSBT);
	context.device.bindBufferMemory(sbtBuffer, sbtMemory, 0);

	void* mapped = context.device.mapMemory(sbtMemory, 0, sbtSize);
	memset(mapped, 0, sbtSize);
	memcpy(static_cast<char*>(mapped), shaderHandles.data(), handleSize);
	memcpy(static_cast<char*>(mapped) + missOffset, shaderHandles.data() + handleSize, handleSize);
	memcpy(static_cast<char*>(mapped) + hitOffset, shaderHandles.data() + 2 * handleSize, handleSize);
	context.device.unmapMemory(sbtMemory);
	vk::DeviceAddress sbtAddress = context.device.getBufferAddress(vk::BufferDeviceAddressInfo{ sbtBuffer }, context.dynamicLoader);

	vk::StridedDeviceAddressRegionKHR raygenRegion{};
	raygenRegion.deviceAddress = sbtAddress;
	raygenRegion.stride = handleStride;
	raygenRegion.size = handleStride;

	vk::StridedDeviceAddressRegionKHR missRegion{};
	missRegion.deviceAddress = sbtAddress + missOffset;
	missRegion.stride = handleStride;
	missRegion.size = handleStride;

	vk::StridedDeviceAddressRegionKHR hitRegion{};
	hitRegion.deviceAddress = sbtAddress + hitOffset;
	hitRegion.stride = handleStride;
	hitRegion.size = handleStride;

	vk::StridedDeviceAddressRegionKHR callableRegion{};

	//Test---------------------
	context.device.waitForFences(context.fence, true, UINT64_MAX);
	context.device.resetFences(context.fence);
	cmd.reset();
	vk::CommandBufferBeginInfo beginInfoThree(vk::CommandBufferUsageFlagBits::eOneTimeSubmit);
	cmd.begin(beginInfoThree);

	cmd.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, context.raytraceExt.mainRayPipeline);

	cmd.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, context.raytraceExt.mainRayLayout, 0, descriptorSet, {});

	cmd.traceRaysKHR(raygenRegion, missRegion, hitRegion, callableRegion, context.currentExtent.width, context.currentExtent.height, 1, context.dynamicLoader);

	cmd.end();
	context.queue.submit(submitInfo, context.fence);

	//---------------------------

	std::cout << "Finished loading Ray ext!" << std::endl;

	for (auto& sampler : model.samplers) {
		vk::SamplerCreateInfo samplerCreate(vk::SamplerCreateFlags{});
		// TODO sampler correct
		out.sampler.push_back(context.device.createSampler(samplerCreate));
	}

	vk::DescriptorPoolSize descPoolSizes(vk::DescriptorType::eCombinedImageSampler, 2 * model.materials.size());
	vk::DescriptorPoolCreateInfo descPoolCreateInfo({}, model.materials.size(), descPoolSizes);
	out.imageDescPool = context.device.createDescriptorPool(descPoolCreateInfo);
	std::vector descSetsToUse(model.materials.size(), context.textureLayout);
	vk::DescriptorSetAllocateInfo descallocateInfo(out.imageDescPool, descSetsToUse);
	out.imageSets = context.device.allocateDescriptorSets(descallocateInfo);
	size_t descriptorID = 0;
	for (auto& material : model.materials) {
		auto& texInfo = material.pbrMetallicRoughness.baseColorTexture;
		vk::DescriptorImageInfo descImageInfo({}, {}, vk::ImageLayout::eShaderReadOnlyOptimal);
		vk::WriteDescriptorSet descWriteSet(out.imageSets[descriptorID], 0, 0, vk::DescriptorType::eCombinedImageSampler, descImageInfo);
		const char* systemUsed = "";
		if (texInfo.index != -1) {
			auto& albedoTex = model.textures[texInfo.index];
			descImageInfo.imageView = out.imageViews[albedoTex.source];
			descImageInfo.sampler = out.sampler[albedoTex.sampler];
			context.device.updateDescriptorSets(descWriteSet, {});
			systemUsed = "USE_ALBEDO";
		}
		auto& normalTexInfo = material.normalTexture;
		if (normalTexInfo.index != -1) {
			auto& normalTex = model.textures[normalTexInfo.index];
			descImageInfo.imageView = out.imageViews[normalTex.source];
			descImageInfo.sampler = out.sampler[normalTex.sampler];
			descWriteSet.dstBinding = 1;
			context.device.updateDescriptorSets(descWriteSet, {});
			systemUsed = "USE_ALB_NORM";
		}
		out.pipes.push_back(createPipeline(context, material.doubleSided, systemUsed, config.fillMode));
		descriptorID++;
	}

	const auto cameraDesc = context.cameraNode.descriptors.back();

	size_t frameBufferID = 0;
	for (auto cCmd : out.cmd) {
		const auto framebuffer = context.framebuffer[frameBufferID++];
		inharitance.setFramebuffer(framebuffer);
		cCmd.begin(cmdBeginInfo);
		size_t cNodeID = 0;
		for (auto& node : model.nodes) {
			if (node.mesh < 0) continue;
			auto& mesh = model.meshes[node.mesh];
			const auto currentModelDesc = out.modelBuffer.descriptors[cNodeID++];

			for (auto& primitive : mesh.primitives) {
				cCmd.bindPipeline(vk::PipelineBindPoint::eGraphics, out.pipes[primitive.material]);
				const MaterialPushConst materialPConst{
					.materialID = (uint32_t)primitive.material,
					.alphaCutoff = (float)model.materials[primitive.material].alphaCutoff };
				cCmd.pushConstants<MaterialPushConst>(context.pipelineLayout, vk::ShaderStageFlagBits::eAllGraphics, 0u, materialPConst);
				std::array descSetsUsed = { currentModelDesc, cameraDesc, out.imageSets[primitive.material] };
				cCmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, context.pipelineLayout,
					0, descSetsUsed, {});

				const auto& indexAccess = model.accessors[primitive.indices];
				cCmd.bindIndexBuffer(out.buffers[indexAccess.bufferView], indexAccess.byteOffset,
					indexAccess.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? vk::IndexType::eUint16 : vk::IndexType::eUint32);

				const auto [posId, posOff] = getFromInput(model, primitive, "POSITION");
				const auto [normalId, normalOff] = getFromInput(model, primitive, "NORMAL");
				const auto [uv1Id, uv1Off] = getFromInput(model, primitive, "TEXCOORD_0");

				std::array buffer = { out.buffers[posId], out.buffers[normalId], out.buffers[uv1Id] };
				std::array offsets = { posOff, normalOff, uv1Off };
				cCmd.bindVertexBuffers(0, buffer, offsets);

				cCmd.drawIndexed(indexAccess.count, 1, 0, 0, 0);
			}

		}
		cCmd.end();
	}

	context.device.waitForFences(context.fence, true, UINT64_MAX);
	context.device.resetFences(context.fence);
	context.device.destroy(pool);

	std::cout << "Loaded " << path << std::endl;
	return true;
}
