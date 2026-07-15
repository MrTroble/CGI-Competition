#version 460
#extension GL_EXT_ray_tracing : require
#extension GL_EXT_nonuniform_qualifier : require

#include "../main/Shared.hpp"

layout(location = 1) rayPayloadInEXT ShadowPayload shadowPayload;

void main()
{
   shadowPayload.isOccluded = true;
   return;
}