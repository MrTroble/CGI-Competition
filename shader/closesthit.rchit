#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "../main/Shared.hpp"

layout(location = 0) rayPayloadInEXT vec3 payload;

// Upper Bound
layout(binding=0, set=2) uniform sampler2D textures[256];

hitAttributeEXT vec2 attribs;

layout(binding=1, set = 2) readonly buffer InstanceOffsetBlock
{
  InstanceOffsets[] offsets;
} I;

layout(binding=2, set = 2) readonly buffer MaterialsBlock
{
  MaterialData[] material;
} M;

layout(binding=3, set = 2) readonly buffer GeometryDataBlock
{
  GeometryData[] geometry;
} G;

layout(binding=4, set = 2) readonly buffer UVBlock
{
  vec2[] uvs;
} UV;

layout(binding=5, set = 2) readonly buffer UIDBlock
{
  uint[] ids;
} ID;

void main()
{
    InstanceOffsets off = I.offsets[gl_InstanceID];
    GeometryData geo = G.geometry[gl_GeometryIndexEXT + off.geometryOffset];
    if(geo.materialID < 0) {
       payload = vec3(1.0, 0.0, 1.0);
       return;
    }
    MaterialData mat = M.material[geo.materialID + off.materialOffset];
    if(mat.texID < 0) {
        payload = vec3(1.0, 0.0, 0.0);
        return;
    }
    const uint index = ((gl_PrimitiveID + geo.offsetIndexLocal) * 3) + off.primitiveOffset;
    const uvec3 currentIds = uvec3(ID.ids[index + 1], ID.ids[index + 2], ID.ids[index + 3]) + uvec3(geo.offsetVertexLocal);
    const vec3 bary = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec2 uvCur = bary.x * UV.uvs[currentIds.x] + bary.y * UV.uvs[currentIds.y] + bary.z * UV.uvs[currentIds.z];
    vec4 color = texture(textures[nonuniformEXT(mat.texID + off.textureOffset)], uvCur);
    payload = color.xyz;
}