#version 460
#extension GL_EXT_ray_tracing : require

#include "../main/Shared.hpp"

layout(location = 0) rayPayloadInEXT Payload payload;

void main()
{
    payload.spectrum = vec3(0.2, 0.4, 1.0); // Blau
}