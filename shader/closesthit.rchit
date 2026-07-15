#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "../main/Shared.hpp"

layout(location = 0) rayPayloadInEXT Payload payload;

layout(location = 1) rayPayloadEXT ShadowPayload shadowPayload;

layout(set=1, binding=0) uniform accelerationStructureEXT topLevelAS;

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

layout(binding=6, set = 2) uniform lightSources
{
  LightSource[NUMBER_OF_LIGHTS] lights;
  uint numberOfLights;

  //for us useless
  uint numberOfLightsWithShadowmap;
} L;

layout(binding=7, set = 2) readonly buffer NormalBlock
{
  vec4[] normal;
} NORM;

#define M_PI 3.1415926535897932384626433832795

vec3 getDirection(vec2 rand) {
    float e1 = 2 * M_PI * rand.x;
    float e2 = acos(1 - rand.y);

    return vec3(cos(e1) * cos(e2), sin(e1) * cos(e2), sin(e2));
}

void main()
{
    InstanceOffsets off = I.offsets[gl_InstanceID];
    GeometryData geo = G.geometry[gl_GeometryIndexEXT + off.geometryOffset];
    if(geo.materialID < 0) {
       payload.spectrum = vec3(1.0, 0.0, 1.0);
       return;
    }
    MaterialData mat = M.material[geo.materialID + off.materialOffset];
    if(mat.texID < 0) {
        payload.spectrum = vec3(1.0, 0.0, 0.0);
        return;
    }
    const uint index = gl_PrimitiveID * 3 + geo.offsetIndexLocal + off.primitiveOffset;
    const uvec3 currentIds = uvec3(ID.ids[index], ID.ids[index + 1], ID.ids[index + 2]) + uvec3(geo.offsetVertexLocal);
    const vec3 bary = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec2 uvCur = bary.x * UV.uvs[currentIds.x] + bary.y * UV.uvs[currentIds.y] + bary.z * UV.uvs[currentIds.z];
    vec4 color = texture(textures[nonuniformEXT(mat.texID + off.textureOffset)], uvCur);
    const vec3 normalCur = normalize(bary.x * NORM.normal[currentIds.x].xyz + bary.y * NORM.normal[currentIds.y].xyz + bary.z * NORM.normal[currentIds.z].xyz);

    const float lambert = max(0, dot(-gl_WorldRayDirectionEXT, normalCur));
    payload.contribution = color.xyz * lambert;

    if(payload.depth < 3) {
        payload.depth++;
        vec3 newOrigin = gl_WorldRayDirectionEXT * gl_HitTEXT + gl_WorldRayOriginEXT + normalCur * 0.01f;
        vec3 newDirection = normalize(getDirection(rngRndVec2(payload.random)));
        //traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, 0, 0, 0, newOrigin, 0.01f, newDirection, 1000.0f, 0);
    }
}