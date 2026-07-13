#ifndef INT_SHARED_IM
#define INT_SHARED_IM

#ifndef INT_GLSL
using uint = unsigned int;
#endif

struct MaterialData {
    int texID;
    int samplerID;
};

struct GeometryData {
    int materialID;
};

struct InstanceOffsets {
    uint geometryOffset;
    uint materialOffset;
    uint textureOffset;
    uint samplerOffset;
};

#endif