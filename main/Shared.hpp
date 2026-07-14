#ifndef INT_SHARED_IM
#define INT_SHARED_IM

#define NUMBER_OF_LIGHTS 64
#ifndef INT_GLSL
#include <glm/glm.hpp>



using uint = unsigned int;
using vec3 = glm::vec3;
using uvec2 = glm::uvec2;
#endif

struct RngShader {
	uvec2 seed;
};

#ifdef INT_GLSL
float uintToFloat(uint x) {
	uint a = 0x3f800000;
	uint u = a | (x >> 9);

	return uintBitsToFloat(u) - 1.f;
}

void rngInit(out RngShader rng, uint i, uint j, uint offset, uint seed) {
	uvec2 state = uvec2(i, j) ^ uvec2(offset << 16, (seed + offset) << 16);
	rng.seed = state;
}

// returns random float between (0,1]
float rngRndFloat(inout RngShader rng) {
	// PCG2D, as described here: https://jcgt.org/published/0009/03/02/
	rng.seed = 1664525u * rng.seed + 1013904223u;
	rng.seed.x += 1664525u * rng.seed.y;
	rng.seed.y += 1664525u * rng.seed.x;
	rng.seed ^= (rng.seed >> 16u);
	rng.seed.x += 1664525u * rng.seed.y;
	rng.seed.y += 1664525u * rng.seed.x;
	rng.seed ^= (rng.seed >> 16u);
	// return float from seed
	return 1.f - uintToFloat(rng.seed.x);
}

// returns random vec2 between (0,1]^2
vec2 rngRndVec2(inout RngShader rng) {
	// PCG2D, as described here: https://jcgt.org/published/0009/03/02/
	rng.seed = 1664525u * rng.seed + 1013904223u;
	rng.seed.x += 1664525u * rng.seed.y;
	rng.seed.y += 1664525u * rng.seed.x;
	rng.seed ^= (rng.seed >> 16u);
	rng.seed.x += 1664525u * rng.seed.y;
	rng.seed.y += 1664525u * rng.seed.x;
	rng.seed ^= (rng.seed >> 16u);
	// return float from seed
	return 1.f - vec2(uintToFloat(rng.seed.x), uintToFloat(rng.seed.y));
}

// returns random vec3 between (0,1]^3
vec3 rngRndVec3(inout RngShader rng) {
	return vec3(rngRndFloat(rng), rngRndVec2(rng));
}
#endif

struct MaterialData {
	int texID;
	int samplerID;
};

struct GeometryData {
	int materialID;
	uint offsetIndexLocal;
	uint offsetVertexLocal;
};

struct InstanceOffsets {
	uint geometryOffset;
	uint materialOffset;
	uint textureOffset;
	uint samplerOffset;
	uint primitiveOffset;
	uint indexOffset;
};

struct Payload
{
	RngShader random;
	vec3 contribution;
	vec3 spectrum;
};

struct LightSource {
	vec3 position;
	float padding1;
	vec3 direction;
	float padding2;
	vec3 color;
	// [0, 1]
	float brightness;
};

#endif