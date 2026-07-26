#include <chrono>
#include <iostream>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>
#include "dnx_generated.h"

#include "MainContext.hpp"
#include "Loader.hpp"
#include "Utilities.hpp"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "lights.hpp"
#include "trackball.hpp"
#include "rng.hpp"
#include "Shared.hpp"

struct UNetLoader {
    std::vector<vk::DeviceMemory> memories;
    std::vector<vk::Buffer> buffer;
    std::vector<vk::ShaderModule> shader;
    vk::DescriptorPool pool;

    std::vector<vk::Pipeline> pipelines;
    std::vector<vk::PipelineLayout> layouts;
    std::vector<std::vector<vk::DescriptorSetLayout>> descLayouts;
    std::vector<std::vector<vk::DescriptorSet>> descSets;
};

template<class T>
inline T resolveDNXScalar(const denox::dnx::ScalarLiteral* literal) {
    // TODO Check correct
    using namespace denox::dnx;
    const ScalarType type = literal->dtype();
    switch (type)
    {
    case ScalarType_I16:
        return (T) * ((const int16_t*)(literal->bytes()->data()));
    case ScalarType_F16:
        return (T) * ((const uint16_t*)(literal->bytes()->data()));
    case ScalarType_U16:
        return (T) * ((const uint16_t*)(literal->bytes()->data()));
    case ScalarType_I32:
        return (T) * ((const int32_t*)(literal->bytes()->data()));
    case ScalarType_F32:
        return (T) * ((const float*)(literal->bytes()->data()));
    case ScalarType_U32:
        return (T) * ((const uint32_t*)(literal->bytes()->data()));
    case ScalarType_I64:
        return (T) * ((const int64_t*)(literal->bytes()->data()));
    case ScalarType_F64:
        return (T) * ((const double*)(literal->bytes()->data()));
    case ScalarType_U64:
        return (T) * ((const uint64_t*)(literal->bytes()->data()));
    default:
        break;
    }
    return *((const T*)(literal->bytes()->data()));
}

template<class T>
inline T resolveSymIR(const std::vector<int64_t>& data, const denox::dnx::SymRef* symRef) {
    return (T)data[symRef->sid()];
}

inline int64_t evaluate(denox::dnx::SymIROpCode opCode, int64_t left, int64_t right) {
    switch (opCode)
    {
    case denox::dnx::SymIROpCode_NOP:
        return 0; // TODO Check
    case denox::dnx::SymIROpCode_ADD:
        return left + right;
    case denox::dnx::SymIROpCode_SUB:
        return left - right;
    case denox::dnx::SymIROpCode_MUL:
        return left * right;
    case denox::dnx::SymIROpCode_DIV:
        return left / right;
    case denox::dnx::SymIROpCode_MOD:
        return left - right * floor(left / right);
    case denox::dnx::SymIROpCode_MIN:
        return std::min(left, right);
    case denox::dnx::SymIROpCode_MAX:
        return std::max(left, right);
    case denox::dnx::SymIROpCode_RHSC:
    case denox::dnx::SymIROpCode_LHSC:
    default:
        throw std::runtime_error("Operation not supported!");
    }
}

#define RESOLVE_LOCAL(obj, func, type) const auto literal##func = obj->##func##_as_literal();\
const auto c##func = literal##func == nullptr ? resolveSymIR<type>(values, obj->##func##_as_symbolic()) :\
    resolveDNXScalar<type>(literal##func);

inline std::vector<const char*> enableUNetReqs(MainContext& context, void** pNext) {
    using namespace denox::dnx;

    const std::string path = (fs::path("..") / "assets" / "net.dnx").string();
    const auto modelFile = loadFile(path);
    const auto model = GetModel(modelFile.data());

    const auto& mInfo = *model->info();
    const auto& compInfo = *mInfo.compilation_info();

    const auto& featComp = *compInfo.features();

    static vk::PhysicalDeviceCooperativeMatrixFeaturesKHR coopMat;
    coopMat.cooperativeMatrix = true;
    static vk::PhysicalDeviceVulkan11Features f11;
    f11.storageBuffer16BitAccess = true;
    f11.storageInputOutput16 = true;
    static vk::PhysicalDeviceVulkan12Features f12;
    f12.shaderFloat16 = true;
    f12.bufferDeviceAddress = true;
    f12.bufferDeviceAddressCaptureReplay = true;
    f12.vulkanMemoryModel = true;
    f12.setShaderSampledImageArrayNonUniformIndexing(true);
    f11.pNext = &f12;
    static vk::PhysicalDeviceSubgroupSizeControlFeatures subgroupFeat;
    f12.pNext = &subgroupFeat;
    subgroupFeat.subgroupSizeControl = true;
    *pNext = &f11;
    pNext = &subgroupFeat.pNext;

    std::vector<const char*> exts;
    exts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    if (featComp.coopmat()) {
        *pNext = &coopMat;
        pNext = &coopMat.pNext;
        exts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
    return exts;
}

inline UNetLoader loadUNet(MainContext& context) {
    using namespace denox::dnx;
    UNetLoader loader;

    const std::string path = (fs::path("..") / "assets" / "net.dnx").string();
    const auto modelFile = loadFile(path);
    const auto model = GetModel(modelFile.data());
    assert(model);

    const auto& sym_ir = *model->sym_ir();
    const auto& ops = *sym_ir.ops();
    const std::array variables = { (int)context.currentExtent.width, (int)context.currentExtent.height };

    std::vector<int64_t> values(
        sym_ir.var_count() + ops.size());
    assert(sym_ir.var_count() == variables.size());

    for (size_t i = 0; i < sym_ir.var_count(); ++i) {
        values[i] = variables[i];
    }

    for (size_t i = 0; i < ops.size(); ++i) {
        const SymIROp op = *ops[i];

        const SymIROpCode base =
            (SymIROpCode)(op.opcode() & ~(SymIROpCode_LHSC | SymIROpCode_RHSC));

        const bool lhs_is_constant =
            (op.opcode() & SymIROpCode_LHSC) != 0;

        const bool rhs_is_constant =
            (op.opcode() & SymIROpCode_RHSC) != 0;

        const int64_t lhs =
            lhs_is_constant
            ? op.lhs()
            : values[static_cast<size_t>(op.lhs())];

        const int64_t rhs =
            rhs_is_constant
            ? op.rhs()
            : values[static_cast<size_t>(op.rhs())];

        values[sym_ir.var_count() + i] =
            evaluate(base, lhs, rhs);
    }
    const auto& buffers = *model->buffers();

    for (const auto buffer : buffers) {
        RESOLVE_LOCAL(buffer, size, vk::DeviceSize);
        const auto size = alignUp(csize, buffer->alignment());

        vk::BufferCreateInfo bufferCreate{ {}, size, vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eStorageTexelBuffer };
        const auto currentBuffer = context.device.createBuffer(bufferCreate);
        loader.buffer.push_back(currentBuffer);

        const auto requ = context.device.getBufferMemoryRequirements(currentBuffer);

        vk::MemoryAllocateInfo allocateInfo{ alignUp(requ.size, requ.alignment), context.hostVisibleDeviceLocalMemory };
        const auto mem = context.device.allocateMemory(allocateInfo);
        loader.memories.push_back(mem);
        context.device.bindBufferMemory(currentBuffer, mem, 0);
    }

    const auto& tensorInits = *model->initializers();
    const auto& tensors = *model->tensors();
    for (const auto init : tensorInits) {
        const auto tensor = tensors[init->tensor()];
        const auto mem = loader.memories[tensor->buffer()];
        RESOLVE_LOCAL(tensor, offset, vk::DeviceSize);
        RESOLVE_LOCAL(tensor, size, vk::DeviceSize);
        void* dest = context.device.mapMemory(mem, coffset, csize);
        const auto& vec = *init->data();
        assert(vec.size() >= csize);
        memcpy(dest, vec.data(), csize);
        context.device.unmapMemory(mem);
    }

    const auto& binaries = *model->shader_binaries();
    for (const auto bin : binaries) {
        const auto& spir = bin->spirv();
        vk::ShaderModuleCreateInfo shaderModCreate({}, spir->size() * sizeof(uint32_t), spir->data());
        const auto shaderMod = context.device.createShaderModule(shaderModCreate);
        loader.shader.push_back(shaderMod);
    }

    vk::DescriptorPoolSize descSize(vk::DescriptorType::eStorageBuffer, 100);
    loader.pool = context.device.createDescriptorPool(vk::DescriptorPoolCreateInfo({}, 100, descSize));

    const auto& dispatches = *model->dispatches();
    vk::PipelineShaderStageRequiredSubgroupSizeCreateInfo groupSize(32u);
    for (const auto disp : dispatches) {
        vk::PipelineShaderStageCreateInfo shaderStageCreateInfo({},
            vk::ShaderStageFlagBits::eCompute, loader.shader[disp->binary_id()], disp->entry_point()->c_str());
        shaderStageCreateInfo.pNext = &groupSize;
        const auto& bindings = *disp->bindings();
        std::vector<vk::DescriptorSetLayout> layoutsCreated;
        for (const auto bind : bindings) {
            std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
            const auto& localBind = *bind->bindings();
            for (const auto lB : localBind) {
                const auto& tensor = *tensors[lB->tensor()];
                vk::DescriptorSetLayoutBinding layoutBinding(lB->binding(), vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute);
                layoutBindings.push_back(layoutBinding);
            }
            vk::DescriptorSetLayoutCreateInfo descrip({}, layoutBindings);
            layoutsCreated.push_back(context.device.createDescriptorSetLayout(descrip));
        }
        loader.descLayouts.push_back(layoutsCreated);
        const auto& pConst = *disp->push_constant();
        vk::PushConstantRange range(vk::ShaderStageFlagBits::eCompute, 0, pConst.size());
        vk::PipelineLayoutCreateInfo pipeLayouteCreateInfo({}, layoutsCreated, range);
        const auto layout = context.device.createPipelineLayout(pipeLayouteCreateInfo);
        loader.layouts.push_back(layout);
        vk::ComputePipelineCreateInfo computePipeCreate({}, shaderStageCreateInfo, layout);
        const auto pipe = context.device.createComputePipeline({}, computePipeCreate);
        loader.pipelines.push_back(pipe.value);

        const auto descSet = context.device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(loader.pool, layoutsCreated));
        loader.descSets.push_back(descSet);
        
        size_t counter = 0;
        for (const auto bind : bindings) {
            std::vector<vk::DescriptorSetLayoutBinding> layoutBindings;
            const auto& localBind = *bind->bindings();
            vk::DescriptorBufferInfo bufferInfo;
            vk::WriteDescriptorSet descSetWrite(descSet[counter], 0, 0, vk::DescriptorType::eStorageBuffer, {}, bufferInfo);
            for (const auto lB : localBind) {
                const auto tensor = tensors[lB->tensor()];
                RESOLVE_LOCAL(tensor, offset, vk::DeviceSize);
                RESOLVE_LOCAL(tensor, size, vk::DeviceSize);
                bufferInfo.buffer = loader.buffer[tensor->buffer()];
                bufferInfo.offset = coffset;
                bufferInfo.range = csize;
                context.device.updateDescriptorSets(descSetWrite, {});
            }
            vk::DescriptorSetLayoutCreateInfo descrip({}, layoutBindings);
            layoutsCreated.push_back(context.device.createDescriptorSetLayout(descrip));
            counter++;
        }
    }
    return loader;
}

inline void notifyViewMatrixChanged(MainContext& context, bool applyJitter = true) {
    const float aspect = (float)context.currentExtent.width / context.currentExtent.height;
    const auto perspective = glm::perspective(glm::radians(60.0f), aspect, 0.1f, 1000.0f);

    // Unit vectors in viewport coordinates
    const auto x1 = glm::vec3(1.0f, 0.0f, 0.0f);
    const auto x2 = glm::vec3(0.0f, 1.0f, 0.0f);

    // Convert to world coordinates
    const glm::mat3 inverse = glm::inverse(context.trackball.view);
    const auto x1World = inverse * x1;
    const auto x2World = inverse * x2;

    // Sample a random jitter vector
    const auto jitter = rngRndFloat(context.rng) * x1World + rngRndFloat(context.rng) * x2World;

    // Apply jitter to view matrix
    auto jitteredViewMatrix = context.trackball.view;
    if (context.settings.useTAA) {
        jitteredViewMatrix = glm::translate(jitteredViewMatrix, jitter * 0.0001f);
    }
    context.cameraNodeData.mvp =
        perspective * jitteredViewMatrix;

    context.cameraNodeData.previousViewMatrix = perspective * context.previousViewMatrix;
    context.previousViewMatrix = jitteredViewMatrix;

    context.cameraNodeData.invView = glm::inverse(jitteredViewMatrix);
    context.cameraNodeData.invProj = glm::inverse(perspective);
    context.cameraNodeData.frame_Index = 0;

    context.cameraNodeData.inverseVP = glm::inverse(context.cameraNodeData.mvp);
}

inline void createSwapchain(MainContext& context) {
    const auto supportedPresentationModes = context.usedPhyDevice.getSurfacePresentModesKHR(context.surface);
    vk::PresentModeKHR presentationMode = vk::PresentModeKHR::eFifo;
    if (std::find(supportedPresentationModes.begin(), supportedPresentationModes.end(), vk::PresentModeKHR::eMailbox) != supportedPresentationModes.end()) {
        presentationMode = vk::PresentModeKHR::eMailbox;
    }

    vk::SwapchainCreateInfoKHR swapchainCreateInfo({}, context.surface, 5, context.formatUsed.format, context.formatUsed.colorSpace,
        context.currentExtent, 1, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eInputAttachment
        | vk::ImageUsageFlagBits::eTransferSrc | vk::ImageUsageFlagBits::eTransferDst, vk::SharingMode::eExclusive, context.queueIndex,
        vk::SurfaceTransformFlagBitsKHR::eIdentity, vk::CompositeAlphaFlagBitsKHR::eOpaque, presentationMode);
    context.swapchain = context.device.createSwapchainKHR(swapchainCreateInfo);
    context.swapchainImages = context.device.getSwapchainImagesKHR(context.swapchain);

    for (auto oldview : context.swapchainImageViews) context.device.destroy(oldview);
    context.swapchainImageViews.clear();
    for (auto oldbuffer : context.framebuffer) context.device.destroy(oldbuffer);
    context.framebuffer.clear();

    vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);
    vk::ImageSubresourceRange depthRange(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1);

    // Setup images for gbuffer
    // Depth
    vk::MemoryAllocateInfo memoryAlloc(0u, context.deviceLocalOnly);
    vk::ImageCreateInfo depthCreateInfo({}, vk::ImageType::e2D, vk::Format::eD16Unorm, vk::Extent3D{ context.currentExtent.width,  context.currentExtent.height, 1u }
        , 1u, 1u, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eDepthStencilAttachment
        | vk::ImageUsageFlagBits::eInputAttachment | vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferSrc, vk::SharingMode::eExclusive, {});
    const auto depthImage = context.device.createImage(depthCreateInfo);
    context.depthImage = depthImage;
    const auto depthImgRequ = context.device.getImageMemoryRequirements(depthImage);
    memoryAlloc.allocationSize = depthImgRequ.size;
    const auto depthImgMemory = context.device.allocateMemory(memoryAlloc);
    context.device.bindImageMemory(depthImage, depthImgMemory, 0);
    context.depthView = context.device.createImageView(vk::ImageViewCreateInfo({}, depthImage, vk::ImageViewType::e2D, vk::Format::eD16Unorm, {}, depthRange));

    // Normal + material
    vk::ImageCreateInfo normalCreateInfo({}, vk::ImageType::e2D, vk::Format::eR16G16B16A16Sfloat, vk::Extent3D{ context.currentExtent.width,  context.currentExtent.height, 1u }
    , 1u, 1u, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eInputAttachment | vk::ImageUsageFlagBits::eSampled, vk::SharingMode::eExclusive, {});
    const auto normalImage = context.device.createImage(normalCreateInfo);
    context.normalImage1 = normalImage;
    const auto normalImgRequ = context.device.getImageMemoryRequirements(normalImage);
    memoryAlloc.allocationSize = normalImgRequ.size;
    const auto normalImgMemory = context.device.allocateMemory(memoryAlloc);
    context.device.bindImageMemory(normalImage, normalImgMemory, 0);
    vk::ImageViewCreateInfo normalViewCreate({}, normalImage, vk::ImageViewType::e2D, vk::Format::eR16G16B16A16Sfloat, {}, range);
    const auto normalImageView = context.device.createImageView(normalViewCreate);
    context.normalView = normalImageView;

    const auto normalImage2 = context.device.createImage(normalCreateInfo);
    const auto normalImgRequ2 = context.device.getImageMemoryRequirements(normalImage2);
    memoryAlloc.allocationSize = normalImgRequ2.size;
    const auto normalImgMemory2 = context.device.allocateMemory(memoryAlloc);
    context.device.bindImageMemory(normalImage2, normalImgMemory2, 0);
    normalViewCreate.image = normalImage2;
    const auto normalImageView2 = context.device.createImageView(normalViewCreate);
    context.normalView2 = normalImageView2;

    // Albedo
    vk::ImageCreateInfo albedoCreateInfo({}, vk::ImageType::e2D, vk::Format::eB8G8R8A8Unorm, vk::Extent3D{ context.currentExtent.width,  context.currentExtent.height, 1u }
    , 1u, 1u, vk::SampleCountFlagBits::e1, vk::ImageTiling::eOptimal, vk::ImageUsageFlagBits::eColorAttachment | vk::ImageUsageFlagBits::eInputAttachment, vk::SharingMode::eExclusive, {});
    const auto albedoImage = context.device.createImage(albedoCreateInfo);
    const auto albedoImgRequ = context.device.getImageMemoryRequirements(albedoImage);
    memoryAlloc.allocationSize = albedoImgRequ.size;
    const auto albedoImgMemory = context.device.allocateMemory(memoryAlloc);
    context.device.bindImageMemory(albedoImage, albedoImgMemory, 0);
    const auto albedoImageView = context.device.createImageView(vk::ImageViewCreateInfo({}, albedoImage, vk::ImageViewType::e2D, vk::Format::eB8G8R8A8Unorm, {}, range));
    context.albedoView = albedoImageView;

    {   // Resolve attachments
        albedoCreateInfo.usage |= vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eTransferDst | vk::ImageUsageFlagBits::eStorage
            | vk::ImageUsageFlagBits::eTransferSrc;
        albedoCreateInfo.setFormat(vk::Format::eR8G8B8A8Unorm);
        context.resolve1Image = context.device.createImage(albedoCreateInfo);
        context.resolve2Image = context.device.createImage(albedoCreateInfo);
        const auto resolve1ImgRequ = context.device.getImageMemoryRequirements(context.resolve1Image);
        memoryAlloc.allocationSize = albedoImgRequ.size;
        const auto resolve1ImgMemory = context.device.allocateMemory(memoryAlloc);
        const auto resolve2ImgMemory = context.device.allocateMemory(memoryAlloc);
        context.device.bindImageMemory(context.resolve1Image, resolve1ImgMemory, 0);
        context.device.bindImageMemory(context.resolve2Image, resolve2ImgMemory, 0);
        const auto resolve1ImageView = context.device.createImageView(vk::ImageViewCreateInfo({}, context.resolve1Image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {}, range));
        const auto resolve2ImageView = context.device.createImageView(vk::ImageViewCreateInfo({}, context.resolve2Image, vk::ImageViewType::e2D, vk::Format::eR8G8B8A8Unorm, {}, range));
        context.resolveView = resolve1ImageView;
        context.resolveView2 = resolve2ImageView;
    }

    for (auto image : context.swapchainImages) {
        vk::ImageViewCreateInfo imageViewCreate({}, image, vk::ImageViewType::e2D, context.formatUsed.format, {}, range);
        const auto imageView = context.device.createImageView(imageViewCreate);
        context.swapchainImageViews.push_back(imageView);
    }

    std::size_t index = 0;
    for (auto imageView : context.swapchainImageViews) {
        const auto normalImageForFrame = index % 2 == 0 ? normalImageView : normalImageView2;
        const auto normalImageOld = index % 2 == 0 ? normalImageView2 : normalImageView;
        const auto oldFrame = index % 2 == 0 ? context.resolveView : context.resolveView2;
        const auto resolveViewUsed = index % 2 == 0 ? context.resolveView2 : context.resolveView;
        const std::array imageViews = { resolveViewUsed, context.depthView, albedoImageView, normalImageForFrame, oldFrame, normalImageOld, imageView };

        const vk::FramebufferCreateInfo frameBufferCreateInfo({}, context.renderPass, imageViews, context.currentExtent.width, context.currentExtent.height, 1);
        context.framebuffer.push_back(context.device.createFramebuffer(frameBufferCreateInfo));

        index += 1;
    }
}

inline void createCommandBuffer(MainContext& context) {
    vk::CommandPoolCreateInfo commandPool(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, context.queueIndex);
    context.commandPool = context.device.createCommandPool(commandPool);

    vk::CommandBufferAllocateInfo allocateInfo(context.commandPool, {}, (uint32_t)context.swapchainImages.size());
    context.command = context.device.allocateCommandBuffers(allocateInfo);

    context.signalSemaphore = context.device.createSemaphore({});
    context.waitSemaphore = context.device.createSemaphore({});
    context.fence = context.device.createFence({});
}

inline void createRenderPass(MainContext& context) {
    const auto surfaceFormats = context.usedPhyDevice.getSurfaceFormatsKHR(context.surface);
    size_t i = 0;
    for (; i < surfaceFormats.size(); i++)
    {
        if (surfaceFormats[i].format == vk::Format::eB8G8R8A8Unorm)
            break;
    }
    context.formatUsed = surfaceFormats[i];
    std::cout << "Used surface format: " << vk::to_string(context.formatUsed.format) << std::endl;

    using namespace vk;

    const std::array attachments = {
        // Resolve
        AttachmentDescription(
            {}, vk::Format::eR8G8B8A8Unorm, SampleCountFlagBits::e1, AttachmentLoadOp::eClear,
            AttachmentStoreOp::eStore, AttachmentLoadOp::eDontCare,
            AttachmentStoreOp::eDontCare, ImageLayout::eUndefined,
            ImageLayout::eGeneral),
            // depth
            AttachmentDescription(
                {}, vk::Format::eD16Unorm, SampleCountFlagBits::e1, AttachmentLoadOp::eClear,
                AttachmentStoreOp::eStore, AttachmentLoadOp::eDontCare, AttachmentStoreOp::eDontCare,
                ImageLayout::eUndefined, ImageLayout::eGeneral),
            // albedo
            AttachmentDescription(
                {}, vk::Format::eB8G8R8A8Unorm, SampleCountFlagBits::e1, AttachmentLoadOp::eClear,
                AttachmentStoreOp::eStore, AttachmentLoadOp::eDontCare, AttachmentStoreOp::eDontCare,
                ImageLayout::eUndefined, ImageLayout::eGeneral),
            // normal + material
            AttachmentDescription(
                {}, vk::Format::eR16G16B16A16Sfloat, SampleCountFlagBits::e1, AttachmentLoadOp::eClear,
                AttachmentStoreOp::eStore, AttachmentLoadOp::eDontCare, AttachmentStoreOp::eDontCare,
                ImageLayout::eUndefined, ImageLayout::eGeneral),
            // Previous out attachment
            AttachmentDescription(
                {}, vk::Format::eR8G8B8A8Unorm, SampleCountFlagBits::e1, AttachmentLoadOp::eLoad,
                AttachmentStoreOp::eStore, AttachmentLoadOp::eDontCare,
                AttachmentStoreOp::eDontCare, ImageLayout::eGeneral,
                ImageLayout::eGeneral),
            // Previous normal + material
            AttachmentDescription(
                    {}, vk::Format::eR16G16B16A16Sfloat, SampleCountFlagBits::e1, AttachmentLoadOp::eLoad,
                    AttachmentStoreOp::eStore, AttachmentLoadOp::eDontCare, AttachmentStoreOp::eDontCare,
                    ImageLayout::eGeneral, ImageLayout::eGeneral),
            // output
            AttachmentDescription(
                {}, context.formatUsed.format, SampleCountFlagBits::e1, AttachmentLoadOp::eClear,
                AttachmentStoreOp::eStore, AttachmentLoadOp::eDontCare,
                AttachmentStoreOp::eDontCare, ImageLayout::eUndefined,
                ImageLayout::eGeneral)
    };

    constexpr std::array firstPassAttachments = {
        AttachmentReference(2, ImageLayout::eGeneral),
        AttachmentReference(3, ImageLayout::eGeneral)
    };
    constexpr AttachmentReference depthAttachmentRef = AttachmentReference(1, ImageLayout::eDepthStencilAttachmentOptimal);

    constexpr std::array inputPassAttachments = {
        AttachmentReference(2, ImageLayout::eGeneral),
        AttachmentReference(3, ImageLayout::eGeneral),
        AttachmentReference(1, ImageLayout::eGeneral)
    };

    constexpr std::array inputTAAPassAttachments = {
        AttachmentReference(1, ImageLayout::eGeneral),
        AttachmentReference(3, ImageLayout::eGeneral),
        AttachmentReference(4, ImageLayout::eGeneral),
        AttachmentReference(5, ImageLayout::eGeneral),
        AttachmentReference(0, ImageLayout::eGeneral)
    };

    constexpr std::array finalColorOut = {
        AttachmentReference(6, ImageLayout::eColorAttachmentOptimal),
        AttachmentReference(0, ImageLayout::eGeneral)
    };

    constexpr std::array finalColorOutUI = {
        AttachmentReference(6, ImageLayout::eColorAttachmentOptimal)
    };

    // G-Buffer pass
    vk::SubpassDescription subpass0({}, vk::PipelineBindPoint::eGraphics, {}, firstPassAttachments, {}, &depthAttachmentRef);
    // Shading pass
    vk::SubpassDescription subpass1({}, vk::PipelineBindPoint::eGraphics, inputPassAttachments, finalColorOut);
    // TAA
    vk::SubpassDescription subpassTaa({}, vk::PipelineBindPoint::eGraphics, inputTAAPassAttachments, finalColorOutUI);
    // Gui Pass
    vk::SubpassDescription subpass2({}, vk::PipelineBindPoint::eGraphics, {}, finalColorOutUI);

    std::array subpasses = { subpass0, subpass1, subpassTaa, subpass2 };
    std::array dependencies = {
        SubpassDependency(0, 1, vk::PipelineStageFlagBits::eAllGraphics, vk::PipelineStageFlagBits::eAllGraphics,
                    AccessFlagBits::eColorAttachmentWrite, AccessFlagBits::eColorAttachmentWrite | AccessFlagBits::eInputAttachmentRead),
        SubpassDependency(1, 2, vk::PipelineStageFlagBits::eAllGraphics, vk::PipelineStageFlagBits::eAllGraphics,
                    AccessFlagBits::eColorAttachmentWrite | AccessFlagBits::eInputAttachmentRead, AccessFlagBits::eColorAttachmentWrite | AccessFlagBits::eShaderRead | AccessFlagBits::eInputAttachmentRead),
        SubpassDependency(2, 3, vk::PipelineStageFlagBits::eAllGraphics, vk::PipelineStageFlagBits::eAllGraphics,
                    AccessFlagBits::eColorAttachmentWrite | AccessFlagBits::eShaderRead | AccessFlagBits::eInputAttachmentRead, AccessFlagBits::eColorAttachmentWrite),
        SubpassDependency(3, VK_SUBPASS_EXTERNAL, vk::PipelineStageFlagBits::eAllGraphics, vk::PipelineStageFlagBits::eAllGraphics,
                    AccessFlagBits::eColorAttachmentWrite, AccessFlagBits::eColorAttachmentWrite)
    };
    vk::RenderPassCreateInfo renderPassCreate({}, attachments, subpasses, dependencies);
    context.renderPass = context.device.createRenderPass(renderPassCreate);
}

inline void createDescriptorSetLayout(MainContext& context) {
    const auto modelBinding = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eAll);
    std::array bindings = { modelBinding };
    context.cameraLayout = context.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, bindings));
    context.modelLayout = context.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, bindings));

    const auto tex1 = vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);
    const auto tex2 = vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment);
    std::array bindingsTex = { tex1, tex2 };
    context.textureLayout = context.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, bindingsTex));

    vk::PushConstantRange pushConstRange(vk::ShaderStageFlagBits::eAllGraphics, 0, sizeof(MaterialPushConst));
    std::array descLayouts = { context.modelLayout, context.cameraLayout, context.textureLayout };
    context.pipelineLayout = context.device.createPipelineLayout(vk::PipelineLayoutCreateInfo({}, descLayouts, pushConstRange));
}

inline void initGUI(MainContext& context) {
    const std::array pool_sizes = {
    vk::DescriptorPoolSize{vk::DescriptorType::eSampler, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eSampledImage, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageImage, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformTexelBuffer, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageTexelBuffer, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBuffer, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBuffer, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eStorageBufferDynamic, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eUniformBufferDynamic, 1000},
    vk::DescriptorPoolSize{vk::DescriptorType::eInputAttachment, 1000} };

    const vk::DescriptorPoolCreateInfo pool_info(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet, (uint32_t)(1000 * pool_sizes.size()), pool_sizes);
    const auto guipool = context.device.createDescriptorPool(pool_info);


    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;

    ImGui_ImplGlfw_InitForVulkan(context.window, true);


    ImGui_ImplVulkan_InitInfo init_info = {};
    init_info.ApiVersion = VK_API_VERSION_1_3;
    init_info.Instance = context.instance;
    init_info.PhysicalDevice = context.usedPhyDevice;
    init_info.Device = context.device;
    init_info.QueueFamily = context.queueIndex;
    init_info.Queue = context.queue;
    init_info.DescriptorPool = guipool;
    init_info.MinImageCount = 3;
    init_info.ImageCount = (uint32_t)context.swapchainImages.size();
    //init_info.PipelineInfoMain.RenderPass = context.renderPass;
    init_info.UseDynamicRendering = true;
    init_info.PipelineInfoMain.Subpass = 3;
    init_info.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    std::array formats{ context.formatUsed.format };
    vk::PipelineRenderingCreateInfo pipeRenderingInfo(0u, formats);
    init_info.PipelineInfoMain.PipelineRenderingCreateInfo = pipeRenderingInfo;
    //init_info.CheckVkResultFn = check_vk_result;
    if (!ImGui_ImplVulkan_Init(&init_info))
        throw std::runtime_error("WELL UI ISSUES!");
}

inline void shutdownGUI() {
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
}

void cleanupSwapChain(MainContext& context) {
    for (auto imageview : context.swapchainImageViews) {
        context.device.destroy(imageview);
    }
    context.swapchainImageViews.clear();

    context.device.destroy(context.depthView);
    context.depthView = nullptr;

    context.device.destroy(context.normalView);
    context.normalView = nullptr;

    context.device.destroy(context.normalView2);
    context.normalView2 = nullptr;

    context.device.destroy(context.albedoView);
    context.albedoView = nullptr;

    context.device.destroy(context.swapchain);
    context.swapchain = nullptr;
}

void updateRayImageSet(MainContext& context) {
    uint32_t zahl = 0;
    for (auto& imageSet : context.raytraceExt.imageSet) {
        vk::WriteDescriptorSet writeAS{};
        writeAS.dstBinding = 0;
        writeAS.descriptorType = vk::DescriptorType::eStorageImage;
        vk::DescriptorImageInfo imageInfo{ {}, context.resolveView, vk::ImageLayout::eGeneral };
        writeAS.setDstSet(imageSet);
        writeAS.setImageInfo(imageInfo);
        context.device.updateDescriptorSets(writeAS, {});
        zahl++;
    }

}

void recreateSwapChain(MainContext& context) {
    std::cout << "Cleanup old swapchain" << std::endl;
    context.device.waitIdle();

    cleanupSwapChain(context);
    std::cout << "Recreating swapchain" << std::endl;
    createSwapchain(context);
    updateRayImageSet(context);

}



// Computes the contents of a single shadow map
void renderShadowMap(MainContext& context, const LightSource& light, std::span<const OutputModel> models, vk::Image target) {
    const auto imageID = 0;

    // Update view matrix for said shadow map
    const auto oldMatrix = context.trackball.view;
    context.trackball.view = glm::lookAt(light.position, light.position + light.direction, glm::vec3(0.0, -1.0, 0.0));
    notifyViewMatrixChanged(context, false);
    context.trackball.view = oldMatrix;

    std::array<vk::ClearValue, 7> clearValues;
    clearValues[1].depthStencil.depth = 1.0f;

    const auto cmd = context.command.back();

    const vk::CommandBufferBeginInfo begin(vk::CommandBufferUsageFlags{});
    cmd.begin(begin);

    constexpr vk::AttachmentReference depthAttachmentRef = vk::AttachmentReference(0, vk::ImageLayout::eDepthStencilAttachmentOptimal);
    const std::array<vk::RenderingAttachmentInfo, 0> colorAttachments = {};

    const vk::RenderPassBeginInfo beginInfo(context.renderPass, context.framebuffer[imageID], { {0,0}, context.currentExtent }, clearValues);
    cmd.beginRenderPass(beginInfo, vk::SubpassContents::eSecondaryCommandBuffers);

    for (auto& model : models) {
        cmd.executeCommands(model.cmd[imageID]);
    }

    cmd.nextSubpass(vk::SubpassContents::eInline);
    // TAA
    cmd.nextSubpass(vk::SubpassContents::eInline);
    // GUI
    cmd.nextSubpass(vk::SubpassContents::eInline);

    cmd.endRenderPass();

    vk::ImageSubresourceLayers subresource(vk::ImageAspectFlagBits::eDepth, 0, 0, 1);
    vk::ImageCopy imageCopy(subresource, {}, subresource, {}, vk::Extent3D(context.currentExtent, 1.0f));
    cmd.copyImage(
        context.depthImage,
        vk::ImageLayout::eGeneral,
        target,
        vk::ImageLayout::eGeneral,
        imageCopy
    );
    cmd.end();

    vk::SubmitInfo submitInfo({}, {}, cmd, {});
    context.queue.submit(submitInfo, context.fence);

    context.device.waitForFences(context.fence, true, UINT64_MAX);
    context.device.resetFences(context.fence);

}

void renderShadowMaps(MainContext& context, std::span<const OutputModel> models) {
    // Run render pass for each image
    glm::mat4* outputPtr = (glm::mat4*)context.device.mapMemory(context.shadowMapContext.shadowMapMatrices, 0, VK_WHOLE_SIZE);

    for (std::size_t i = 0; i < context.numberOfLightsWithShadowMaps; i++) {
        std::cout << "Rendering shadow maps: " << i + 1 << "/" << context.numberOfLightsWithShadowMaps << std::endl;
        const auto& light = context.lights[i];
        renderShadowMap(context, light, models, context.shadowMapContext.shadowMaps[i]);
        outputPtr[i] = context.cameraNodeData.mvp;
    }

    context.device.unmapMemory(context.shadowMapContext.shadowMapMatrices);
}

inline void createRaytracingPipeline(MainContext& context) {
    std::array shaderStagesLoaded = {
        loadShaderStages(context,  "raygen.rgen.spv"),
        loadShaderStages(context,  "miss.rmiss.spv"),
        loadShaderStages(context, "closesthit.rchit.spv"),
        loadShaderStages(context, "lightHit.rchit.spv"),
        loadShaderStages(context, "shadow.rchit.spv"),
        loadShaderStages(context, "shadowLight.rchit.spv")
    };

    std::array<vk::DescriptorSetLayoutBinding, 4> bindings{};
    bindings[0].binding = 0;
    bindings[0].descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
    bindings[0].descriptorCount = 1;
    bindings[0].stageFlags = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindings[1].binding = 1;
    bindings[1].descriptorType = vk::DescriptorType::eUniformBuffer;
    bindings[1].descriptorCount = 1;
    bindings[1].stageFlags = vk::ShaderStageFlagBits::eRaygenKHR | vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindings[2].binding = 0;
    bindings[2].descriptorType = vk::DescriptorType::eStorageImage;
    bindings[2].descriptorCount = 1;
    bindings[2].stageFlags = vk::ShaderStageFlagBits::eRaygenKHR;

    std::array<vk::DescriptorSetLayoutBinding, 8> bindingsTextures{};
    bindingsTextures[0].binding = 0;
    bindingsTextures[0].descriptorType = vk::DescriptorType::eCombinedImageSampler;
    bindingsTextures[0].descriptorCount = MAX_IMAGE_TEXTURES_RTX;
    bindingsTextures[0].stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindingsTextures[1].binding = 1;
    bindingsTextures[1].descriptorType = vk::DescriptorType::eStorageBuffer;
    bindingsTextures[1].descriptorCount = 1;
    bindingsTextures[1].stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindingsTextures[2].binding = 2;
    bindingsTextures[2].descriptorType = vk::DescriptorType::eStorageBuffer;
    bindingsTextures[2].descriptorCount = 1;
    bindingsTextures[2].stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindingsTextures[3].binding = 3;
    bindingsTextures[3].descriptorType = vk::DescriptorType::eStorageBuffer;
    bindingsTextures[3].descriptorCount = 1;
    bindingsTextures[3].stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindingsTextures[4].binding = 4;
    bindingsTextures[4].descriptorType = vk::DescriptorType::eStorageBuffer;
    bindingsTextures[4].descriptorCount = 1;
    bindingsTextures[4].stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindingsTextures[5].binding = 5;
    bindingsTextures[5].descriptorType = vk::DescriptorType::eStorageBuffer;
    bindingsTextures[5].descriptorCount = 1;
    bindingsTextures[5].stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindingsTextures[6].binding = 6;
    bindingsTextures[6].descriptorType = vk::DescriptorType::eUniformBuffer;
    bindingsTextures[6].descriptorCount = 1;
    bindingsTextures[6].stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    bindingsTextures[7].binding = 7;
    bindingsTextures[7].descriptorType = vk::DescriptorType::eStorageBuffer;
    bindingsTextures[7].descriptorCount = 1;
    bindingsTextures[7].stageFlags = vk::ShaderStageFlagBits::eClosestHitKHR | vk::ShaderStageFlagBits::eAnyHitKHR;

    vk::DescriptorSetLayoutCreateInfo desriptorLayoutInfo{ {}, 2, bindings.data() };
    context.raytraceExt.descLayout = context.device.createDescriptorSetLayout(desriptorLayoutInfo);
    desriptorLayoutInfo.setBindings(bindings[2]);
    context.raytraceExt.descImageLayout = context.device.createDescriptorSetLayout(desriptorLayoutInfo);
    desriptorLayoutInfo.setBindings(bindingsTextures);
    context.raytraceExt.descTextureLayout = context.device.createDescriptorSetLayout(desriptorLayoutInfo);

    std::array descriptorLayoutsLocal = { context.raytraceExt.descImageLayout,
                            context.raytraceExt.descLayout, context.raytraceExt.descTextureLayout };
    vk::PipelineLayoutCreateInfo pipelineLayoutInfo{ {}, descriptorLayoutsLocal };
    context.raytraceExt.mainRayLayout = context.device.createPipelineLayout(pipelineLayoutInfo);

    vk::RayTracingShaderGroupCreateInfoKHR raygenGroup{};
    raygenGroup.setType(vk::RayTracingShaderGroupTypeKHR::eGeneral);
    raygenGroup.setGeneralShader(0);

    vk::RayTracingShaderGroupCreateInfoKHR missGroup{};
    missGroup.setType(vk::RayTracingShaderGroupTypeKHR::eGeneral);
    missGroup.setGeneralShader(1);

    vk::RayTracingShaderGroupCreateInfoKHR hitGroup{};
    hitGroup.setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
    hitGroup.setClosestHitShader(2);
    hitGroup.setGeneralShader(2);

    vk::RayTracingShaderGroupCreateInfoKHR hitGroupLight{};
    hitGroupLight.setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
    hitGroupLight.setClosestHitShader(3);
    hitGroupLight.setGeneralShader(3);

    vk::RayTracingShaderGroupCreateInfoKHR hitGroupShadow{};
    hitGroupShadow.setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
    hitGroupShadow.setClosestHitShader(4);

    vk::RayTracingShaderGroupCreateInfoKHR hitGroupShadowLight{};
    hitGroupShadowLight.setType(vk::RayTracingShaderGroupTypeKHR::eTrianglesHitGroup);
    hitGroupShadowLight.setClosestHitShader(5);

    std::array<vk::RayTracingShaderGroupCreateInfoKHR, 6> groups = { raygenGroup, missGroup, hitGroup, hitGroupLight, hitGroupShadow,hitGroupShadowLight };

    vk::RayTracingPipelineCreateInfoKHR pipelineInfo{};
    pipelineInfo.setStages(shaderStagesLoaded);
    pipelineInfo.setGroups(groups);
    pipelineInfo.setMaxPipelineRayRecursionDepth(3);
    pipelineInfo.setLayout(context.raytraceExt.mainRayLayout);

    auto pipeline = context.device.createRayTracingPipelineKHR({}, {}, pipelineInfo, nullptr, context.dynamicLoader);
    context.raytraceExt.mainRayPipeline = pipeline.value;

    vk::DescriptorPoolSize imagePoolSize{
        vk::DescriptorType::eStorageImage,
        2
    };

    constexpr size_t MAX_POOL = 100;

    std::array mainPoolInfos = { vk::DescriptorPoolSize{
        vk::DescriptorType::eAccelerationStructureKHR,
        MAX_POOL
    }, vk::DescriptorPoolSize{
        vk::DescriptorType::eUniformBuffer,
        MAX_POOL
    }, vk::DescriptorPoolSize{
        vk::DescriptorType::eCombinedImageSampler,
        MAX_IMAGE_TEXTURES_RTX
    } , vk::DescriptorPoolSize{
        vk::DescriptorType::eStorageBuffer,
        MAX_POOL
    } };

    vk::DescriptorPoolCreateInfo poolInfo{ {}, MAX_POOL, mainPoolInfos };
    context.raytraceExt.pool = context.device.createDescriptorPool(poolInfo);

    vk::DescriptorPoolCreateInfo poolInfoImage{ {}, imagePoolSize.descriptorCount, imagePoolSize };
    context.raytraceExt.imagePool = context.device.createDescriptorPool(poolInfoImage);

    std::array imageDescsLocal = { context.raytraceExt.descImageLayout , context.raytraceExt.descImageLayout };
    vk::DescriptorSetAllocateInfo imageAllocate{ context.raytraceExt.imagePool, imageDescsLocal };
    context.raytraceExt.imageSet = context.device.allocateDescriptorSets(imageAllocate);
    updateRayImageSet(context);

    vk::PhysicalDeviceRayTracingPipelinePropertiesKHR rayTracingProperties{};
    vk::PhysicalDeviceProperties2 properties{};
    properties.pNext = &rayTracingProperties;
    context.usedPhyDevice.getProperties2(&properties);
    uint32_t handleSize = rayTracingProperties.shaderGroupHandleSize;


    //todo remove magic number
    uint32_t groupCount = 6;
    std::vector<uint8_t> shaderHandles(groupCount * handleSize);
    auto handels = context.device.getRayTracingShaderGroupHandlesKHR(context.raytraceExt.mainRayPipeline, 0, groupCount, shaderHandles.size(), shaderHandles.data(), context.dynamicLoader);

    vk::BufferUsageFlags usage = vk::BufferUsageFlagBits::eShaderBindingTableKHR | vk::BufferUsageFlagBits::eShaderDeviceAddress | vk::BufferUsageFlagBits::eTransferDst;
    uint32_t handleAlignment = rayTracingProperties.shaderGroupHandleAlignment;
    uint32_t baseAlignment = rayTracingProperties.shaderGroupBaseAlignment;

    uint32_t hitGroupCount = 4;
    uint32_t handleStride = alignUp(handleSize, handleAlignment);
    uint32_t raygenSize = handleStride;
    uint32_t missSize = handleStride;
    uint32_t hitSize = handleStride * hitGroupCount;
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
    for (uint32_t i = 0; i < hitGroupCount; i++)
    {
        memcpy(
            static_cast<char*>(mapped) + hitOffset + i * handleStride,
            shaderHandles.data() + (2 + i) * handleSize,
            handleSize
        );
    }
    context.device.unmapMemory(sbtMemory);

    vk::DeviceAddress sbtAddress = context.device.getBufferAddress(vk::BufferDeviceAddressInfo{ sbtBuffer }, context.dynamicLoader);

    context.raytraceExt.raygenRegion.deviceAddress = sbtAddress;
    context.raytraceExt.raygenRegion.stride = handleStride;
    context.raytraceExt.raygenRegion.size = handleStride;

    context.raytraceExt.missRegion.deviceAddress = sbtAddress + missOffset;
    context.raytraceExt.missRegion.stride = handleStride;
    context.raytraceExt.missRegion.size = handleStride;

    context.raytraceExt.hitRegion.deviceAddress = sbtAddress + hitOffset;
    context.raytraceExt.hitRegion.stride = handleStride;
    context.raytraceExt.hitRegion.size = hitSize;
}

template<class T>
inline size_t expand(std::vector<T>& toExpand, std::span<const T> values) {
    const auto sIndex = toExpand.size();
    toExpand.resize(sIndex + values.size());
    std::copy(values.begin(), values.end(), toExpand.begin() + sIndex);
    return sIndex;
}

inline void createTLA(MainContext& context, std::span<const OutputModel> outModels) {
    vk::BufferCreateInfo tlasInstanceBufferInfo{};
    tlasInstanceBufferInfo.size = sizeof(vk::AccelerationStructureInstanceKHR) * outModels.size();
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
    void* data = context.device.mapMemory(instanceMemory, 0, VK_WHOLE_SIZE);
    for (size_t i = 0; i < outModels.size(); i++)
    {
        memcpy((vk::AccelerationStructureInstanceKHR*)data + i,
            &outModels[i].rayHolder.instanceInfo, sizeof(vk::AccelerationStructureInstanceKHR));
    }
    context.device.unmapMemory(instanceMemory);
    vk::DeviceAddress instanceAddress = context.device.getBufferAddress({ instanceBuffer }, context.dynamicLoader);

    vk::AccelerationStructureGeometryKHR tlasGeometry{};
    tlasGeometry.geometryType = vk::GeometryTypeKHR::eInstances;
    tlasGeometry.geometry.instances = { false, instanceAddress };
    tlasGeometry.flags = vk::GeometryFlagBitsKHR::eOpaque;

    vk::AccelerationStructureBuildGeometryInfoKHR tlasBuildInfo{};
    tlasBuildInfo.type = vk::AccelerationStructureTypeKHR::eTopLevel;
    tlasBuildInfo.flags = vk::BuildAccelerationStructureFlagBitsKHR::ePreferFastTrace;
    tlasBuildInfo.mode = vk::BuildAccelerationStructureModeKHR::eBuild;
    tlasBuildInfo.setGeometries(tlasGeometry);

    uint32_t maxAmounts = outModels.size();
    auto tlasSize = context.device.getAccelerationStructureBuildSizesKHR(vk::AccelerationStructureBuildTypeKHR::eDevice, tlasBuildInfo, maxAmounts, context.dynamicLoader);

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
    context.raytraceExt.topLevelStructure = context.device.createAccelerationStructureKHR(tlasCreateInfo, nullptr, context.dynamicLoader);
    tlasBuildInfo.dstAccelerationStructure = context.raytraceExt.topLevelStructure;

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
    tlasBuildInfo.scratchData = context.device.getBufferAddress({ tlasScratch }, context.dynamicLoader);

    vk::DescriptorSetAllocateInfo allocInfo{ context.raytraceExt.pool, context.raytraceExt.descLayout };
    context.raytraceExt.topLevelDescSet = context.device.allocateDescriptorSets(allocInfo)[0];

    vk::WriteDescriptorSetAccelerationStructureKHR asInfo{};
    asInfo.setAccelerationStructures(context.raytraceExt.topLevelStructure);

    vk::WriteDescriptorSet writeAS{};
    writeAS.dstBinding = 0;
    writeAS.descriptorCount = 1;
    writeAS.descriptorType = vk::DescriptorType::eAccelerationStructureKHR;
    writeAS.pNext = &asInfo;
    writeAS.setDstSet(context.raytraceExt.topLevelDescSet);
    context.device.updateDescriptorSets(writeAS, {});

    vk::DescriptorBufferInfo bufferInfo(context.cameraNode.buffer, 0, sizeof(context.cameraNodeData));
    writeAS.setBufferInfo(bufferInfo);
    writeAS.dstBinding = 1;
    writeAS.descriptorType = vk::DescriptorType::eUniformBuffer;
    context.device.updateDescriptorSets(writeAS, {});

    vk::CommandPoolCreateInfo poolCreateInfo(vk::CommandPoolCreateFlagBits::eResetCommandBuffer, context.queueIndex);
    const auto pool = context.device.createCommandPool(poolCreateInfo);
    vk::CommandBufferAllocateInfo allocateInfo(pool, vk::CommandBufferLevel::ePrimary, 1);
    const auto cmd = context.device.allocateCommandBuffers(allocateInfo).back();
    vk::CommandBufferBeginInfo cmdBeginInfo(vk::CommandBufferUsageFlags{});

    context.raytraceExt.textureSet = context.device.allocateDescriptorSets({ context.raytraceExt.pool,
        context.raytraceExt.descTextureLayout }).back();
    std::vector<vk::DescriptorImageInfo> imageInfos;

    std::vector<vk::Sampler> samplerToDesc;
    std::vector<vk::ImageView> viewsToDesc;
    std::vector<MaterialData> materialData;
    std::vector<GeometryData> geometryData;

    cmd.begin(cmdBeginInfo);

    std::vector<InstanceOffsets> startIndexArray;
    std::array<size_t, 3> amountCurrent = { 0 };
    for (auto& model : outModels)
    {
        amountCurrent[0] += model.completeAmountToCopy;
        amountCurrent[1] += model.completeAmountToIndexCopy;
        amountCurrent[2] += model.completeAmountToNormalCopy;
    }
    {
        for (size_t i = 0; i < context.raytraceExt.additionalBuffer.size(); i++)
        {
            const auto usage = vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer;
            const auto buffer = context.device.createBuffer(vk::BufferCreateInfo{
                {},
                (uint32_t)amountCurrent[i],
                usage,
                vk::SharingMode::eExclusive
                });
            const auto bufferRequ = context.device.getBufferMemoryRequirements(buffer);
            vk::MemoryAllocateInfo memoryAlloc(bufferRequ.size, context.deviceLocalOnly);
            const auto memory = context.device.allocateMemory(memoryAlloc);
            context.device.bindBufferMemory(buffer, memory, 0);
            context.raytraceExt.additionalBuffer[i] = { memory, buffer };
        }

        amountCurrent.fill(0);
        for (auto& model : outModels)
        {
            InstanceOffsets offsetsL;
            offsetsL.textureOffset = expand<vk::ImageView>(viewsToDesc, model.imageViews);
            offsetsL.materialOffset = expand<MaterialData>(materialData, model.rtxMaterials);
            offsetsL.geometryOffset = expand<GeometryData>(geometryData, model.rtxGeodata);
            offsetsL.samplerOffset = expand<vk::Sampler>(samplerToDesc, model.sampler);
            offsetsL.primitiveOffset = amountCurrent[0] / sizeof(glm::vec2);
            offsetsL.indexOffset = amountCurrent[1] / sizeof(uint32_t);
            startIndexArray.push_back(offsetsL);
            std::array spans = { std::span{model.rtxCopyData}, std::span{model.rtxIndexCopyData}, std::span{model.rtxNormalCopyData} };
            size_t currentLocalIndex = 0;
            for (auto& amountMax : amountCurrent)
            {
                const auto currentIndex = currentLocalIndex++;
                for (const auto& copy : spans[currentIndex]) {
                    cmd.copyBuffer(copy.buffer.buffer, context.raytraceExt.additionalBuffer[currentIndex].buffer, vk::BufferCopy{ copy.offset, amountMax, copy.size });
                    amountMax += copy.size;
                }
            }
        }
    }

    vk::AccelerationStructureBuildRangeInfoKHR rangeInfo{ (uint32_t)outModels.size() };
    cmd.buildAccelerationStructuresKHR(tlasBuildInfo, &rangeInfo, context.dynamicLoader);

    cmd.end();

    vk::SubmitInfo submitInfo({}, {}, cmd);
    context.queue.submit(submitInfo);
    context.queue.waitIdle();

    for (auto& model : outModels)
    {
        std::array spans = { std::span{model.rtxCopyData}, std::span{model.rtxIndexCopyData}, std::span{model.rtxNormalCopyData} };
        for (const auto sp : spans) {
            for (const auto& c : sp) {
                if (c.markedForDeletion) {
                    context.device.destroy(c.buffer.buffer);
                    context.device.free(c.buffer.memory);
                }
            }
        }
    }

    context.device.destroy(pool);

    context.raytraceExt.instanceBuffer = createBuffer<InstanceOffsets>(context, startIndexArray, vk::BufferUsageFlagBits::eStorageBuffer);
    context.raytraceExt.geoBuffer = createBuffer<GeometryData>(context, geometryData, vk::BufferUsageFlagBits::eStorageBuffer);
    context.raytraceExt.materialBuffer = createBuffer<MaterialData>(context, materialData, vk::BufferUsageFlagBits::eStorageBuffer);
    vk::DescriptorBufferInfo bufferInfoLocal(context.raytraceExt.instanceBuffer.buffer, 0, VK_WHOLE_SIZE);
    vk::WriteDescriptorSet arrayDescLocal(context.raytraceExt.textureSet, 1, 0, 1, vk::DescriptorType::eStorageBuffer,
        nullptr, &bufferInfoLocal);
    vk::WriteDescriptorSet lightdesLocal(context.raytraceExt.textureSet, 1, 0, 1, vk::DescriptorType::eUniformBuffer,
        nullptr, &bufferInfoLocal);
    context.device.updateDescriptorSets(arrayDescLocal, {});
    bufferInfoLocal.setBuffer(context.raytraceExt.geoBuffer.buffer);
    arrayDescLocal.setDstBinding(3);
    context.device.updateDescriptorSets(arrayDescLocal, {});
    bufferInfoLocal.setBuffer(context.raytraceExt.materialBuffer.buffer);
    arrayDescLocal.setDstBinding(2);
    context.device.updateDescriptorSets(arrayDescLocal, {});
    bufferInfoLocal.setBuffer(context.raytraceExt.additionalBuffer[0].buffer);
    arrayDescLocal.setDstBinding(4);
    context.device.updateDescriptorSets(arrayDescLocal, {});
    bufferInfoLocal.setBuffer(context.raytraceExt.additionalBuffer[1].buffer);
    arrayDescLocal.setDstBinding(5);
    context.device.updateDescriptorSets(arrayDescLocal, {});
    bufferInfoLocal.setBuffer(context.lightSources.buffer);
    lightdesLocal.setDstBinding(6);
    context.device.updateDescriptorSets(lightdesLocal, {});
    bufferInfoLocal.setBuffer(context.raytraceExt.additionalBuffer[2].buffer);
    arrayDescLocal.setDstBinding(7);
    context.device.updateDescriptorSets(arrayDescLocal, {});

    if (!samplerToDesc.empty()) {
        const auto sampler = samplerToDesc[0]; // TODO BETTER WAY
        for (auto& tex : viewsToDesc) {
            imageInfos.emplace_back(sampler, tex, vk::ImageLayout::eShaderReadOnlyOptimal);
        }
        const auto oldSizeC = imageInfos.size();
        if (oldSizeC < MAX_IMAGE_TEXTURES_RTX) {
            imageInfos.resize(MAX_IMAGE_TEXTURES_RTX);
            std::fill(imageInfos.begin() + oldSizeC, imageInfos.end(), imageInfos[0]);
        }
        else {
            if (oldSizeC > MAX_IMAGE_TEXTURES_RTX) {
                std::cerr << "Warning!: Max textures are over the limit!" << std::endl;
            }
        }
        vk::WriteDescriptorSet arrayDesc(context.raytraceExt.textureSet, 0, 0, vk::DescriptorType::eCombinedImageSampler,
            imageInfos);
        context.device.updateDescriptorSets(arrayDesc, {});
    }
}

inline void renderFrame(MainContext& context, std::span<const OutputModel> models) {
    glfwPollEvents();
    if (glfwGetWindowAttrib(context.window, GLFW_ICONIFIED) != 0)
    {
        ImGui_ImplGlfw_Sleep(10);
        return;
    }

    uint32_t imageID;
    const auto acquireResult = vk::Result(vkAcquireNextImageKHR(context.device, context.swapchain, UINT64_MAX, context.waitSemaphore, vk::Fence{}, &imageID));
    if (acquireResult == vk::Result::eSuboptimalKHR || acquireResult == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain(context);
        return;
    }
    else if (acquireResult != vk::Result::eSuccess) {
        throw std::runtime_error("acquiring swapchain image failed");
    }

    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();


    const auto cmd = context.command[imageID];
    std::array<vk::ClearValue, 7> clearValues;
    clearValues[0].color.float32[3] = 1.0f;
    clearValues[1].depthStencil.depth = 1.0f;
    clearValues[6].color.float32[3] = 1.0f;
    const vk::CommandBufferBeginInfo begin(vk::CommandBufferUsageFlags{});
    cmd.begin(begin);
    auto& s = context.settings;

    if (!s.useRayTracing) {
        const vk::RenderPassBeginInfo beginInfo(context.renderPass, context.framebuffer[imageID], { {0,0}, context.currentExtent }, clearValues);
        cmd.beginRenderPass(beginInfo, vk::SubpassContents::eSecondaryCommandBuffers);

        for (auto& model : models) {
            cmd.executeCommands(model.cmd[imageID]);
        }

        cmd.nextSubpass(vk::SubpassContents::eInline);

        cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, context.lightPipeline);

        std::vector<RenderTillingInfo> infos(s.baseTillingInfo.xAmount * s.baseTillingInfo.yAmount * s.baseTillingInfo.zAmount);
        RenderTillingInfo infoTilling = s.baseTillingInfo;
        for (size_t x = 0; x < s.baseTillingInfo.xAmount; x++)
        {
            infoTilling.xOffset = x;
            for (size_t y = 0; y < s.baseTillingInfo.yAmount; y++)
            {
                infoTilling.yOffset = y;
                for (size_t z = 0; z < s.baseTillingInfo.zAmount; z++)
                {
                    infoTilling.zOffset = z;
                    infos[indexFrom({ x, y, z }, s.baseTillingInfo)] = infoTilling;
                }
            }
        }

        rebuildLightLists(context);

        size_t lightListIndex = 0;
        for (const auto& value : infos) {
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, context.lightLayout, 0,
                { context.lightMapSet, context.cameraNode.descriptors[0], context.lightSources.descriptors[0], context.lightListUpload.descriptors[lightListIndex++] }, {});
            cmd.pushConstants<RenderTillingInfo>(context.lightLayout, (vk::ShaderStageFlags)vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0u, value);
            cmd.draw(4, 1, 0, 0);
        }

        // Taa
        cmd.nextSubpass(vk::SubpassContents::eInline);

        if (s.useTAA) {
            cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, context.taaPipeline);
            cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, context.taaLayout, 0, { context.taaSets[imageID], context.cameraNode.descriptors[0] }, {});
            cmd.draw(4, 1, 0, 0);
        }

        // gui
        cmd.nextSubpass(vk::SubpassContents::eInline);
        cmd.endRenderPass();

        if (s.useTAA) {
            const auto targetImage = imageID % 2 == 0 ? context.resolve2Image : context.resolve1Image;
            vk::ImageSubresourceLayers subresource(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
            vk::ImageCopy imageCopy(subresource, {}, subresource, {}, vk::Extent3D(context.currentExtent, 1.0f));
            cmd.copyImage(
                context.swapchainImages[imageID],
                vk::ImageLayout::eGeneral,
                targetImage,
                vk::ImageLayout::eGeneral,
                imageCopy
            );
        }
    }
    else {
        const auto targetImage = context.resolve1Image;
        cmd.bindPipeline(vk::PipelineBindPoint::eRayTracingKHR, context.raytraceExt.mainRayPipeline);
        cmd.bindDescriptorSets(vk::PipelineBindPoint::eRayTracingKHR, context.raytraceExt.mainRayLayout, 0,
            { context.raytraceExt.imageSet[imageID % 2], context.raytraceExt.topLevelDescSet, context.raytraceExt.textureSet }, {});
        cmd.traceRaysKHR(context.raytraceExt.raygenRegion, context.raytraceExt.missRegion, context.raytraceExt.hitRegion,
            context.raytraceExt.callableRegion, context.currentExtent.width, context.currentExtent.height, 1, context.dynamicLoader);
        vk::ImageSubresourceLayers subresource(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
        vk::ImageCopy imageCopy(subresource, {}, subresource, {}, vk::Extent3D(context.currentExtent, 1.0f));
        cmd.copyImage(
            targetImage,
            vk::ImageLayout::eGeneral,
            context.swapchainImages[imageID],
            vk::ImageLayout::eGeneral,
            imageCopy
        );

    }

    vk::RenderingAttachmentInfo attachmentInfo(context.swapchainImageViews[imageID], vk::ImageLayout::eGeneral,
        vk::ResolveModeFlagBits::eNone);
    vk::Rect2D area({}, context.currentExtent);
    vk::RenderingInfo renderingInfo({}, area, 1, 0, attachmentInfo);
    cmd.beginRendering(renderingInfo);

    ImGui::NewFrame();

    ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_None);
    ImGui::Checkbox("Show Lights", &s.showLights);
    ImGui::Checkbox("Use TAA", &s.useTAA);
    ImGui::Checkbox("Show AO", &s.useAO);
    ImGui::Checkbox("Use ray tracing", &s.useRayTracing);
    ImGui::Checkbox("Accumulate", &s.accumulate);
    context.cameraNodeData.windowParameter.z = s.useAO ? 1.0f : 0.0f;

    ImGui::SliderInt3("Tiling", (int*)&s.baseTillingInfo.xAmount, 1, MAX_TILLING);

    ImGui::End();

    //ImGui::ShowDemoWindow();

    ImGui::Render();
    ImGui_ImplVulkan_RenderDrawData(ImGui::GetDrawData(), cmd);
    cmd.endRendering();

    vk::ImageSubresourceRange subRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1);

    vk::ImageMemoryBarrier barrier1(vk::AccessFlagBits::eNone, vk::AccessFlagBits::eTransferWrite,
        vk::ImageLayout::eGeneral, vk::ImageLayout::ePresentSrcKHR, context.queueIndex, context.queueIndex,
        {}, subRange);
    barrier1.image = context.swapchainImages[imageID];
    cmd.pipelineBarrier(vk::PipelineStageFlagBits::eAllGraphics, vk::PipelineStageFlagBits::eAllGraphics, vk::DependencyFlagBits::eByRegion,
        {}, {}, barrier1);

    cmd.end();

    const std::array waitFlags = { vk::PipelineStageFlags(vk::PipelineStageFlagBits::eAllGraphics) };
    vk::SubmitInfo submitInfo(context.waitSemaphore, waitFlags, cmd, context.signalSemaphore);
    context.queue.submit(submitInfo, context.fence);

    const vk::PresentInfoKHR presentInfo(context.signalSemaphore, context.swapchain, imageID,
        nullptr);

    // C API because the bindings are being stupid
    const auto result = vk::Result(vkQueuePresentKHR(context.queue, (VkPresentInfoKHR*)&presentInfo));

    // const auto result = context.queue.presentKHR(presentInfo);
    if (result == vk::Result::eErrorOutOfDateKHR) {
        recreateSwapChain(context);
        return;
    }

    context.device.waitForFences(context.fence, true, UINT64_MAX);
    context.device.resetFences(context.fence);
}

inline void createLightPipe(MainContext& context) {

    std::array shaderStagesLoaded = {
        loadShaderStages(context, "light.vert.spv"),
        loadShaderStages(context, "light.frag.spv")
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputState(vk::PipelineVertexInputStateCreateFlags{});

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleFan, false);

    vk::Viewport viewportIn(0, 0, (float)context.currentExtent.width, (float)context.currentExtent.height, 0, 1.0f);
    vk::Rect2D rect2d({}, context.currentExtent);
    vk::PipelineViewportStateCreateInfo viewport({}, viewportIn, rect2d);

    vk::PipelineRasterizationStateCreateInfo rasterizationCreateInfo({}, false, false, vk::PolygonMode::eFill, {});
    rasterizationCreateInfo.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisampleCreate(vk::PipelineMultisampleStateCreateFlags{});

    vk::PipelineDepthStencilStateCreateInfo depthStencil({}, false, false, vk::CompareOp::eLess);

    std::array<vk::PipelineColorBlendAttachmentState, 2> attachmentCount;
    // TODO configure
    attachmentCount[0].colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;
    attachmentCount[1] = attachmentCount[0];

    vk::PipelineColorBlendStateCreateInfo pipelineColorBlend({}, false, {}, attachmentCount);

    std::array descriptor = { vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eInputAttachment, 1, vk::ShaderStageFlagBits::eFragment),
                              vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eInputAttachment, 1, vk::ShaderStageFlagBits::eFragment),
                              vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eInputAttachment, 1, vk::ShaderStageFlagBits::eFragment),
                              vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment) };
    vk::DescriptorSetLayoutCreateInfo descSetCreateInfo({}, descriptor);
    context.lightDescLayout = context.device.createDescriptorSetLayout(descSetCreateInfo);

    std::array lightSourceDescriptor = {
        vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, NUMBER_OF_BIG_LIGHTS, vk::ShaderStageFlagBits::eFragment),
        vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment),
    };
    context.lightSourceLayout = context.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, lightSourceDescriptor));

    std::array lightListDescriptor = { vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eUniformBuffer, 1, vk::ShaderStageFlagBits::eFragment) };
    context.lightListLayout = context.device.createDescriptorSetLayout(vk::DescriptorSetLayoutCreateInfo({}, lightListDescriptor));

    createLightSources(context);

    std::array sizesForPool = { vk::DescriptorPoolSize(vk::DescriptorType::eInputAttachment, 3),
                                vk::DescriptorPoolSize(vk::DescriptorType::eCombinedImageSampler, 1) };
    vk::DescriptorPoolCreateInfo poolCreate({}, 1, sizesForPool);
    context.lightPool = context.device.createDescriptorPool(poolCreate);
    vk::DescriptorSetAllocateInfo allocateInfo(context.lightPool, context.lightDescLayout);
    context.lightMapSet = context.device.allocateDescriptorSets(allocateInfo).back();

    std::array pipelineLayouts = { context.lightDescLayout, context.cameraLayout, context.lightSourceLayout, context.lightListLayout };
    vk::PushConstantRange pushConstRange(vk::ShaderStageFlagBits::eVertex | vk::ShaderStageFlagBits::eFragment, 0, sizeof(RenderTillingInfo));
    vk::PipelineLayoutCreateInfo layoutCreateInfo({}, pipelineLayouts, pushConstRange);
    context.lightLayout = context.device.createPipelineLayout(layoutCreateInfo);

    vk::GraphicsPipelineCreateInfo pipeCreateInfo({}, shaderStagesLoaded, &vertexInputState,
        &inputAssembly, {}, &viewport, &rasterizationCreateInfo, &multisampleCreate, &depthStencil, &pipelineColorBlend, {},
        context.lightLayout, context.renderPass, 1);
    const auto result = context.device.createGraphicsPipeline({}, pipeCreateInfo);
    context.lightPipeline = result.value;

    vk::DescriptorImageInfo descImageInfo({}, context.albedoView, vk::ImageLayout::eGeneral);
    vk::WriteDescriptorSet writeDescriptorSets(context.lightMapSet, 0, 0, vk::DescriptorType::eInputAttachment, descImageInfo);
    context.device.updateDescriptorSets(writeDescriptorSets, {});
    descImageInfo.imageView = context.normalView;
    writeDescriptorSets.dstBinding = 1;
    context.device.updateDescriptorSets(writeDescriptorSets, {});
    descImageInfo.imageView = context.depthView;
    writeDescriptorSets.dstBinding = 2;
    context.device.updateDescriptorSets(writeDescriptorSets, {});
    writeDescriptorSets.descriptorType = vk::DescriptorType::eCombinedImageSampler;
    writeDescriptorSets.dstBinding = 3;
    descImageInfo.setSampler(context.device.createSampler(vk::SamplerCreateInfo{}));
    context.device.updateDescriptorSets(writeDescriptorSets, {});
}



inline void createTAA(MainContext& context) {

    std::array shaderStagesLoaded = {
        loadShaderStages(context, "taa.vert.spv"),
        loadShaderStages(context, "taa.frag.spv")
    };

    vk::PipelineVertexInputStateCreateInfo vertexInputState(vk::PipelineVertexInputStateCreateFlags{});

    vk::PipelineInputAssemblyStateCreateInfo inputAssembly({}, vk::PrimitiveTopology::eTriangleFan, false);

    vk::Viewport viewportIn(0, 0, (float)context.currentExtent.width, (float)context.currentExtent.height, 0, 1.0f);
    vk::Rect2D rect2d({}, context.currentExtent);
    vk::PipelineViewportStateCreateInfo viewport({}, viewportIn, rect2d);

    vk::PipelineRasterizationStateCreateInfo rasterizationCreateInfo({}, false, false, vk::PolygonMode::eFill, {});
    rasterizationCreateInfo.lineWidth = 1.0f;

    vk::PipelineMultisampleStateCreateInfo multisampleCreate(vk::PipelineMultisampleStateCreateFlags{});

    vk::PipelineDepthStencilStateCreateInfo depthStencil({}, false, false, vk::CompareOp::eLess);

    std::array<vk::PipelineColorBlendAttachmentState, 1> attachmentCount;
    attachmentCount[0] = vk::PipelineColorBlendAttachmentState(true, vk::BlendFactor::eSrcAlpha, vk::BlendFactor::eOneMinusSrcAlpha,
        vk::BlendOp::eAdd, vk::BlendFactor::eZero, vk::BlendFactor::eOne, vk::BlendOp::eAdd);
    attachmentCount[0].colorWriteMask = vk::FlagTraits<vk::ColorComponentFlagBits>::allFlags;
    //attachmentCount[1] = attachmentCount[0];

    vk::PipelineColorBlendStateCreateInfo pipelineColorBlend({}, false, {}, attachmentCount);

    std::array descriptor = { vk::DescriptorSetLayoutBinding(0, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment),
                              vk::DescriptorSetLayoutBinding(1, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment),
                              vk::DescriptorSetLayoutBinding(2, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment),
                              vk::DescriptorSetLayoutBinding(3, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment),
                              vk::DescriptorSetLayoutBinding(4, vk::DescriptorType::eCombinedImageSampler, 1, vk::ShaderStageFlagBits::eFragment) };
    vk::DescriptorSetLayoutCreateInfo descSetCreateInfo({}, descriptor);
    context.taaDescLayout = context.device.createDescriptorSetLayout(descSetCreateInfo);

    vk::DescriptorPoolSize sizesForPool(vk::DescriptorType::eCombinedImageSampler, context.swapchainImages.size() * 5);
    vk::DescriptorPoolCreateInfo poolCreate({}, context.swapchainImages.size(), sizesForPool);
    context.taaPool = context.device.createDescriptorPool(poolCreate);
    std::vector layouts(context.swapchainImages.size(), context.taaDescLayout);
    vk::DescriptorSetAllocateInfo allocateInfo(context.taaPool, layouts);
    context.taaSets = context.device.allocateDescriptorSets(allocateInfo);

    std::array pipelineLayouts = { context.taaDescLayout, context.cameraLayout };
    vk::PipelineLayoutCreateInfo layoutCreateInfo({}, pipelineLayouts);
    context.taaLayout = context.device.createPipelineLayout(layoutCreateInfo);

    vk::GraphicsPipelineCreateInfo pipeCreateInfo({}, shaderStagesLoaded, &vertexInputState,
        &inputAssembly, {}, &viewport, &rasterizationCreateInfo, &multisampleCreate, &depthStencil, &pipelineColorBlend, {},
        context.taaLayout, context.renderPass, 2);
    const auto result = context.device.createGraphicsPipeline({}, pipeCreateInfo);
    context.taaPipeline = result.value;

    const auto samplerCreateInfo = vk::SamplerCreateInfo();
    const auto sampler = context.device.createSampler(samplerCreateInfo);

    for (std::size_t i = 0; i < context.taaSets.size(); i++) {
        const auto set = context.taaSets[i];
        // TODO
        vk::DescriptorImageInfo descImageInfo(sampler, context.albedoView, vk::ImageLayout::eGeneral);
        vk::WriteDescriptorSet writeDescriptorSets(set, 0, 0, vk::DescriptorType::eCombinedImageSampler, descImageInfo);
        size_t oldImageID = (i + 3) % context.swapchainImages.size();

        descImageInfo.imageView = i % 2 == 0 ? context.resolveView : context.resolveView2;
        writeDescriptorSets.dstBinding = 0;
        context.device.updateDescriptorSets(writeDescriptorSets, {});

        descImageInfo.imageView = context.depthView;
        writeDescriptorSets.dstBinding = 1;
        context.device.updateDescriptorSets(writeDescriptorSets, {});

        descImageInfo.imageView = i % 2 == 0 ? context.normalView : context.normalView2;
        writeDescriptorSets.dstBinding = 2;
        context.device.updateDescriptorSets(writeDescriptorSets, {});

        descImageInfo.imageView = i % 2 == 0 ? context.normalView2 : context.normalView;
        writeDescriptorSets.dstBinding = 3;
        context.device.updateDescriptorSets(writeDescriptorSets, {});

        descImageInfo.imageView = i % 2 == 0 ? context.resolveView2 : context.resolveView;
        writeDescriptorSets.dstBinding = 4;
        context.device.updateDescriptorSets(writeDescriptorSets, {});
    }
}


inline GLTFLoader loadModels(MainContext& context, std::vector<OutputModel>& outmodels) {
    GLTFLoader loader;
    outmodels.resize(3 + context.lights.size());
    loader.load(context, "assets/provence.glb", outmodels[0], ModelConfig{ .position = glm::vec3(0.0f) });
    //loader.load(context, "assets/DamagedHelmet.glb", outmodels[0], ModelConfig{ .position = glm::vec3(0.0f, 1.0f, 0.0f) });	
    loader.load(context, "assets/MetalRoughSpheres.glb", outmodels[1], ModelConfig{
        .position = glm::vec3(3.0f, 4.5f, -10.0f), .scale = glm::vec3(-0.2, 0.2, 0.2) });
    //loader.load(context, "assets/IORTestGrid.glb", outmodels[2], ModelConfig{
     // .position = glm::vec3(3.0f, 4.5f, -0.0f), .scale = glm::vec3(2, 2, 2), .rotation = glm::rotate(glm::mat4(1.0f),glm::radians(-90.0f),glm::vec3(1.0f, 0.0f, 0.0f)) });
    loader.load(context, "assets/MosquitoInAmber.glb", outmodels[2], ModelConfig{
  .position = glm::vec3(3.0f, 4.5f, -0.0f), .scale = glm::vec3(0.2, 0.2, 0.2) , .rotation = glm::rotate(glm::mat4(1.0f),glm::radians(90.0f),glm::vec3(1.0f, 0.0f, 0.0f)) });

    std::cout << "Loading " << context.lights.size() << " debug cubes for lights" << std::endl;
    for (std::size_t i = 0; i < context.lights.size(); i++) {
        const auto& light = context.lights[i];
        loader.load(
            context,
            "assets/cube.glb",
            outmodels[3 + i],
            ModelConfig{ .fillMode = vk::PolygonMode::eLine, .position = light.position, .scale = glm::vec3(0.3f),
                         .shaderBindingOffset = 1 },// TODO Change shader binding for RTX
            i
        );
    }
    return loader;
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    auto context = (MainContext*)glfwGetWindowUserPointer(window);

    float setTo = 0.0;
    if (action == GLFW_PRESS) {
        setTo = SPEED;
    }
    else if (action == GLFW_RELEASE) {
        setTo = 0.0;
    }
    else {
        return;
    }

    if (key == GLFW_KEY_A) {
        (*context).trackball.cameraMovement.x = -setTo;
    }
    else if (key == GLFW_KEY_D) {
        (*context).trackball.cameraMovement.x = setTo;
    } if (key == GLFW_KEY_SPACE) {
        (*context).trackball.cameraMovement.y = setTo;
    }
    else if (key == GLFW_KEY_LEFT_SHIFT) {
        (*context).trackball.cameraMovement.y = -setTo;
    } if (key == GLFW_KEY_W) {
        (*context).trackball.cameraMovement.z = -setTo;
    }
    else if (key == GLFW_KEY_S) {
        (*context).trackball.cameraMovement.z = setTo;
    }
    else {
        return;
    }
}

static void mouse_cursor_callback(GLFWwindow* window, double xpos, double ypos)
{
    ImGui_ImplGlfw_CursorPosCallback(window, xpos, ypos);

    auto context = (MainContext*)glfwGetWindowUserPointer(window);
    bool viewMatrixChanged = trackball_drag_update(*context);
    if (viewMatrixChanged) {
        notifyViewMatrixChanged(*context);
    }
}

static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    ImGui_ImplGlfw_MouseButtonCallback(window, button, action, mods);

    if (button != GLFW_MOUSE_BUTTON_LEFT) {
        return;
    }
    auto context = (MainContext*)glfwGetWindowUserPointer(window);

    if (action == GLFW_PRESS) {
        trackball_drag_start(*context);
    }
    else {
        trackball_drag_end(*context);
        notifyViewMatrixChanged(*context);
    }
}

VkBool32 debugMessage(vk::DebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
    vk::DebugUtilsMessageTypeFlagsEXT messageTypes,
    const vk::DebugUtilsMessengerCallbackDataEXT* pCallbackData,
    void* pUserData) {
    switch (messageSeverity) {
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose:
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo:
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning:
        break;
    case vk::DebugUtilsMessageSeverityFlagBitsEXT::eError:
        break;
    default:
        break;
    }

    std::cout << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}

int main()
{
    if (!glfwInit())
    {
        std::cout << "Too bad, glfw can not be initalized!" << std::endl;
        return -1;
    }

    if (!glfwVulkanSupported()) {
        std::cerr << "vulkan not supported" << std::endl;
        return -1;
    }

    uint32_t countExt;
    const char** extensions = glfwGetRequiredInstanceExtensions(&countExt);
    if (extensions == nullptr) {
        std::cout << "Too bad, glfw not working!" << std::endl;
        return -1;
    }

    MainContext context;

    auto extensionToEnable = std::vector(extensions, extensions + countExt);
    extensionToEnable.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    std::vector<const char*> layersToEnable = { "VK_LAYER_KHRONOS_validation" };
    const auto instanceLayer = vk::enumerateInstanceLayerProperties();
    std::vector<const char*> layersEnabled;
    for (const auto name : layersToEnable) {
        const auto iter = std::find_if(instanceLayer.begin(), instanceLayer.end(), [&](const vk::LayerProperties& layer) {
            return std::string(name) == layer.layerName.data(); });
        if (iter == std::end(instanceLayer)) {
            std::cout << "Could not add layer " << name << std::endl;
        }
        else {
            layersEnabled.push_back(name);
        }
    }

    vk::ApplicationInfo appInfo("RayTracer-App", VK_MAKE_VERSION(1, 0, 0),
        "RayTracer-App", VK_MAKE_VERSION(1, 0, 0), VK_API_VERSION_1_3);
    vk::InstanceCreateInfo instanceCreateInfo({}, &appInfo, layersEnabled, extensionToEnable);
    context.instance = vk::createInstance(instanceCreateInfo);



    const vk::DebugUtilsMessengerCreateInfoEXT debugUtilsMsgCreateInfo(
        {},
        (vk::DebugUtilsMessageSeverityFlagsEXT)
        vk::FlagTraits<vk::DebugUtilsMessageSeverityFlagBitsEXT>::allFlags,
        (vk::DebugUtilsMessageTypeFlagsEXT)
        vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral,
        (PFN_vkDebugUtilsMessengerCallbackEXT)debugMessage);
    const auto dynamicLoader = vk::detail::DispatchLoaderDynamic(context.instance, vkGetInstanceProcAddr);
    const auto messenger = context.instance.createDebugUtilsMessengerEXTUnique(debugUtilsMsgCreateInfo, nullptr, dynamicLoader);

    const auto phyDevices = context.instance.enumeratePhysicalDevices();
    // TODO selection
    context.usedPhyDevice = phyDevices[0];

    const auto memProp = context.usedPhyDevice.getMemoryProperties();
    {
        uint32_t count = 0;
        for (auto value : memProp.memoryTypes) {
            if (count >= memProp.memoryTypeCount) break;
            if (value.propertyFlags == vk::MemoryPropertyFlagBits::eDeviceLocal
                && context.deviceLocalOnly == UINT32_MAX) {
                context.deviceLocalOnly = count;
            }
            const vk::MemoryPropertyFlags toIgnore = vk::MemoryPropertyFlagBits::eHostCoherent | vk::MemoryPropertyFlagBits::eHostCached;
            if ((value.propertyFlags & ~toIgnore) == vk::MemoryPropertyFlagBits::eHostVisible) {
                context.hostVisibleOnly = count;
            }
            if (value.propertyFlags & vk::MemoryPropertyFlagBits::eHostVisible
                && value.propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal
                && (uint32_t)value.propertyFlags < (uint32_t)vk::MemoryPropertyFlagBits::eDeviceCoherentAMD
                && context.hostVisibleDeviceLocalMemory == UINT32_MAX) {
                context.hostVisibleDeviceLocalMemory = count;
            }
            if (context.hostVisibleDeviceLocalMemory != UINT32_MAX &&
                context.deviceLocalOnly != UINT32_MAX &&
                context.hostVisibleOnly != UINT32_MAX) {
                break;
            }
            count++;
        }
    }

    const auto properties = context.usedPhyDevice.getProperties();
    std::cout << "Used device is: " << properties.deviceName << std::endl;

    if (context.hostVisibleDeviceLocalMemory == UINT32_MAX) {
        std::cout << "Could not find host visible device local memory falling back to host visible" << std::endl;
        context.hostVisibleDeviceLocalMemory = context.hostVisibleOnly;
    }

    const auto queueFamilies = context.usedPhyDevice.getQueueFamilyProperties();

    const auto queueIter = std::find_if(queueFamilies.begin(), queueFamilies.end(), [](const vk::QueueFamilyProperties& value) {
        return value.queueFlags & vk::QueueFlags(vk::QueueFlagBits::eGraphics); });
    if (queueIter == std::end(queueFamilies)) {
        std::cout << "Well no graphics queue detected ... too bad!" << std::endl;
        return -1;
    }
    context.queueIndex = (uint32_t)std::distance(queueIter, queueFamilies.begin());

    std::array<float, 1> priorities = { 1.0f };
    vk::DeviceQueueCreateInfo deviceQueueCreateInfo({}, context.queueIndex, priorities);

    std::vector<const char*> deviceExtensions = { VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_RAY_TRACING_PIPELINE_EXTENSION_NAME,
                            VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME };
    // VK_KHR_ray_tracing_pipeline_maintainence?
    vk::PhysicalDeviceDynamicRenderingFeatures dynamicFeature(true);
    vk::PhysicalDeviceFeatures2 physicalDeviceFeatures;
    physicalDeviceFeatures.features.fillModeNonSolid = true;
    physicalDeviceFeatures.features.shaderFloat64 = true;
    physicalDeviceFeatures.features.shaderInt16 = true;
    physicalDeviceFeatures.features.shaderInt64 = true;

    vk::PhysicalDeviceRayTracingPipelineFeaturesKHR rayTracingPipelineFeatures{};
    rayTracingPipelineFeatures.rayTracingPipeline = true;
    vk::PhysicalDeviceAccelerationStructureFeaturesKHR accelFeaturesInfo;
    accelFeaturesInfo.accelerationStructure = true;
    accelFeaturesInfo.accelerationStructureCaptureReplay = true;
    accelFeaturesInfo.descriptorBindingAccelerationStructureUpdateAfterBind = true;
    physicalDeviceFeatures.pNext = &dynamicFeature;
    dynamicFeature.pNext = &accelFeaturesInfo;
    accelFeaturesInfo.pNext = &rayTracingPipelineFeatures;
    const auto exts = enableUNetReqs(context, &rayTracingPipelineFeatures.pNext);
    for (const auto e : exts) deviceExtensions.push_back(e);
    vk::DeviceCreateInfo deviceCreateInfo({}, deviceQueueCreateInfo, {}, deviceExtensions);
    deviceCreateInfo.pNext = &physicalDeviceFeatures;

    context.device = context.usedPhyDevice.createDevice(deviceCreateInfo);
    std::cout << "Device created" << std::endl;

    context.dynamicLoader = vk::detail::DispatchLoaderDynamic(context.instance, vkGetInstanceProcAddr, context.device, vkGetDeviceProcAddr);

    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    GLFWmonitor* monitor = glfwGetPrimaryMonitor();
    const GLFWvidmode* mode = glfwGetVideoMode(monitor);

    glfwWindowHint(GLFW_RED_BITS, mode->redBits);
    glfwWindowHint(GLFW_GREEN_BITS, mode->greenBits);
    glfwWindowHint(GLFW_BLUE_BITS, mode->blueBits);
    glfwWindowHint(GLFW_REFRESH_RATE, mode->refreshRate);

    context.currentExtent = vk::Extent2D{ (uint32_t)mode->width / 2, (uint32_t)mode->height / 2 };
    context.window = glfwCreateWindow(context.currentExtent.width, context.currentExtent.height, "Window Title", NULL, NULL);
    glfwSetWindowUserPointer(context.window, &context);

    VkSurfaceKHR surface;
    VkResult err = glfwCreateWindowSurface(context.instance, context.window, NULL, &surface);
    if (err)
    {
        std::cout << "Well could not create surface!" << std::endl;
        return -1;
    }
    context.surface = surface;

    context.queue = context.device.getQueue(context.queueIndex, 0);

    createRenderPass(context);
    createSwapchain(context);
    createCommandBuffer(context);
    createDescriptorSetLayout(context);
    loadUNet(context);

    context.cameraNodeData.windowParameter.x = (float)context.currentExtent.width;
    context.cameraNodeData.windowParameter.y = (float)context.currentExtent.height;
    context.cameraNodeData.windowParameter.z = 0.0f;
    context.cameraNode = createArrayBuffer(context, std::span{ &context.cameraNodeData, 1u }, context.cameraLayout);

    context.trackball.view = glm::lookAt(glm::vec3(-1.0f, 3.0f, 0.0f), glm::vec3(0.0f, 3.0, 0.0), glm::vec3(0.0f, -1.0f, 0.0f));
    notifyViewMatrixChanged(context);

    createLightPipe(context);
    createTAA(context);
    createRaytracingPipeline(context);

    initGUI(context);

    std::cout << "Loaded all pipes" << std::endl << "Begin Model loading" << std::endl;

    std::vector<OutputModel> models;
    loadModels(context, models);
    createTLA(context, models);
    renderShadowMaps(context, std::span{ &models[0], 1u });
    glfwSetKeyCallback(context.window, key_callback);
    glfwSetCursorPosCallback(context.window, mouse_cursor_callback);
    glfwSetMouseButtonCallback(context.window, mouse_button_callback);

    context.trackball.lastPositionUpdate = std::chrono::system_clock::now();
    while (!glfwWindowShouldClose(context.window)) {
        glfwPollEvents();

        if (trackball_move_camera(context)) notifyViewMatrixChanged(context);
        if (context.settings.accumulate) context.cameraNodeData.frame_Index++;
        else context.cameraNodeData.frame_Index = 1;
        updateMemory(context, context.cameraNode.memory, std::span{ (uint8_t*)&context.cameraNodeData, sizeof(context.cameraNodeData) });

        if (context.settings.showLights) {
            renderFrame(context, std::span{ models.begin(), models.size() });
        }
        else {
            renderFrame(context, std::span{ &models[0], 1u });
        }
    }

    shutdownGUI();
    context.instance.destroy(surface);
    glfwDestroyWindow(context.window);
    context.device.destroy();
    context.instance.destroy();
    glfwTerminate();
    return 0;
}
