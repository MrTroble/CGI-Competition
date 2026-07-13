#ifndef INT_SHARED_IM
#define INT_SHARED_IM

#ifndef INT_GLSL
using uint = unsigned int;
#endif

struct Material {
    uint texID;
    uint samplerID;
};

struct GeometryData {
    uint materialID;
};

struct InstanceOffsets {
    uint geometryID;
};

#endif