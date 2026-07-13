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
    std::vector<VkTransformMatrixKHR> transforms;
    size_t perNodeId = 0;
    for (auto& node : model.nodes) {
        PerNodeData& nodeData = nodeMatrix[perNodeId++];
        nodeData.mvp *= matrixFromNode(node);
        for (auto childID : node.children) {
            nodeMatrix[childID].mvp = nodeData.mvp;
        }
        nodeData.previousViewMatrix = glm::transpose(glm::inverse(nodeData.mvp));
        VkTransformMatrixKHR matLocal{
            nodeData.mvp[0][0], nodeData.mvp[1][0], nodeData.mvp[2][0], nodeData.mvp[3][0],
            nodeData.mvp[0][1], nodeData.mvp[1][1], nodeData.mvp[2][1], nodeData.mvp[3][1],
            nodeData.mvp[0][2], nodeData.mvp[1][2], nodeData.mvp[2][2], nodeData.mvp[3][2]
        };
        transforms.push_back(matLocal);
    }
    out.modelBuffer = createArrayBuffer<PerNodeData>(context, nodeMatrix, context.modelLayout);

    const auto usage = vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eUniformBuffer
        | vk::BufferUsageFlagBits::eAccelerationStructureBuildInputReadOnlyKHR;
    const auto bufferTransforms = context.device.createBuffer(vk::BufferCreateInfo{
        {},
        (uint32_t)transforms.size() * sizeof(transforms[0]),
        usage,
        vk::SharingMode::eExclusive
        });
    const auto bufferRequ = context.device.getBufferMemoryRequirements(bufferTransforms);
    const auto memoryTrafo = loadToMemory(context, { (uint8_t*)transforms.data(), transforms.size() * sizeof(transforms[0]) }, bufferRequ);
    context.device.bindBufferMemory(bufferTransforms, memoryTrafo, 0);
    const auto modelDataAdd = context.device.getBufferAddress({ bufferTransforms }, context.dynamicLoader);

    perNodeId = 0;
    for (auto& node : model.nodes) {
        const vk::DeviceAddress currentModel = modelDataAdd + perNodeId * sizeof(VkTransformMatrixKHR);
        perNodeId++;
        if (node.mesh >= 0) {
            auto& mesh = model.meshes[node.mesh];
            for (auto& primitive : mesh.primitives) {
                vk::AccelerationStructureGeometryKHR currentData;
                const auto& indexAccess = model.accessors[primitive.indices];
                currentData.flags = vk::GeometryFlagBitsKHR::eOpaque;
                currentData.geometry.triangles.indexType = (indexAccess.componentType == TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT ? vk::IndexType::eUint16 : vk::IndexType::eUint32);
                const auto addressIndex = context.device.getBufferAddress({ out.buffers[indexAccess.bufferView] }, context.dynamicLoader);
                currentData.geometry.triangles.indexData.deviceAddress = addressIndex + indexAccess.byteOffset;
                currentData.geometry.triangles.vertexFormat = vk::Format::eR32G32B32Sfloat;
                const auto [posId, posOff] = getFromInput(model, primitive, "POSITION");
                const auto addressVertex = context.device.getBufferAddress({ out.buffers[posId] }, context.dynamicLoader);
                currentData.geometry.triangles.vertexData = addressVertex + posOff;
                currentData.geometry.triangles.vertexStride = sizeof(glm::vec3);
                currentData.geometry.triangles.maxVertex = indexAccess.count - 1;
                currentData.geometry.triangles.transformData.deviceAddress = currentModel;
                const uint32_t primitiveSize = (uint32_t)indexAccess.count / 3;
                geometryData.push_back(currentData);
                maxPrimitive.push_back(primitiveSize);
                vk::AccelerationStructureBuildRangeInfoKHR primitveRanges{ primitiveSize, 0, 0, 0 };
                structureBuildRange.push_back(primitveRanges);
                out.rtxGeodata.push_back(GeometryData{primitive.material});
            }
        }
    }
    RaytracingExtHolder extHolder;

    std::array<vk::AccelerationStructureBuildGeometryInfoKHR, 1> geometryInfo = {
        vk::AccelerationStructureBuildGeometryInfoKHR{vk::AccelerationStructureTypeKHR::eBottomLevel}
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
        const auto scratchbufferRequ = context.device.getBufferMemoryRequirements(scratchbuffer);

        allocInfo.allocationSize = alignUp(scratchbufferRequ.size, scratchbufferRequ.alignment);
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
        extHolder.accelerationStructures = dataStructure;
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

    vk::AccelerationStructureDeviceAddressInfoKHR blasAddressInfo{};
    blasAddressInfo.accelerationStructure = extHolder.accelerationStructures;
    vk::DeviceAddress blasAddress = context.device.getAccelerationStructureAddressKHR(blasAddressInfo, context.dynamicLoader);
    VkTransformMatrixKHR identMatrix = {
        1.0f, 0,    0,    0,
        0,    1.0f, 0,    0,
        0,    0,    1.0f, 0
    };
    vk::AccelerationStructureInstanceKHR tlasInstance{};
    tlasInstance.transform = identMatrix;
    tlasInstance.instanceCustomIndex = 0;
    tlasInstance.mask = 0xFF;
    tlasInstance.instanceShaderBindingTableRecordOffset = 0;
    tlasInstance.setFlags(vk::GeometryInstanceFlagBitsKHR::eTriangleFacingCullDisable);
    tlasInstance.accelerationStructureReference = blasAddress;
    extHolder.instanceInfo = tlasInstance;

    out.rayHolder = extHolder;
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
            out.rtxMaterials.push_back(MaterialData{ albedoTex.source, albedoTex.sampler });
        }
        else {
            out.rtxMaterials.push_back(MaterialData{ -1, -1});
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
