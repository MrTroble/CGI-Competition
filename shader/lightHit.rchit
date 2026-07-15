#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "../main/Shared.hpp"

layout(location = 0) rayPayloadInEXT Payload payload;

layout(location = 1) rayPayloadEXT ShadowPayload shadowPayload;

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
void main()
{
    uint lightID = gl_InstanceCustomIndexEXT;
    //if(lightID < L.numberOfLights){
    LightSource light = L.lights[lightID];
    payload.spectrum = light.color * payload.contribution *  light.brightness;
    //}else{
    //    payload.spectrum = vec3(1.0f, 1.0f, 1.0f);
    //}
 
    return;
}