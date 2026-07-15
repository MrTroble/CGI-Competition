#include <oidn.hpp>

struct OIDNInfo {
    size_t width;
    size_t height;
};

void startup(const OIDNInfo& info) {
    oidn::DeviceRef device = oidn::newDevice();
    device.commit();

    // Create buffers for input/output images accessible by both host (CPU) and device (CPU/GPU)
    oidn::BufferRef colorBuf = device.newBuffer(info.width * info.height * 4 * sizeof(float));

        // Create a filter for denoising a beauty (color) image using optional auxiliary images too
        // This can be an expensive operation, so try not to create a new filter for every image!
    oidn::FilterRef filter = device.newFilter("RT"); // generic ray tracing filter
    filter.setImage("color", colorBuf, oidn::Format::Float4, info.width, info.height); // beauty
    filter.setImage("output", colorBuf, oidn::Format::Float4, info.width, info.height); // denoised beauty
    filter.commit();
    //oidnNewSharedBufferFromFD(device, OIDN_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD, );
}