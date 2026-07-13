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
  Material[] material;
} M;

layout(binding=1, set = 2) readonly buffer GeometryDataBlock
{
  GeometryData[] geometry;
} G;

void main()
{
    payload = vec3(1.0, 0.0, 0.0); // Rot
}