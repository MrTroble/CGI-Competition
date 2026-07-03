#version 460

struct LightSource {
	vec3 position;
	float padding1;
	vec3 direction;
	float padding2;
	vec3 color;
	float brightness;
};

layout(input_attachment_index=0, set=0, binding = 0) uniform subpassInput ALBEDO;
layout(input_attachment_index=1, set=0, binding = 1) uniform subpassInput NORMAL;
layout(input_attachment_index=2, set=0, binding = 2) uniform subpassInput DEPTH;
layout(set=0, binding = 3) uniform sampler2D DEPTHINPUT;

layout(binding=0, set=1) uniform CAMERA {
	mat4 mvp;
	mat4 previousViewMatrix;
	mat4 inversVP;
	vec4 windowSize;
} camera;

#define NUMBER_OF_MAX_LIGHTS 64

layout(binding=0, set=2) uniform LIGHT_SOURCES {
	LightSource[NUMBER_OF_MAX_LIGHTS] lights;
	uint numberOfLights;
	uint numberOfLightsWithShadowMaps;
} lightSources;

layout(std140, binding=0, set=3) uniform LIGHT_LIST_IDS {
	uvec4[NUMBER_OF_MAX_LIGHTS / 4] ids;
	layout(offset = 256) uint numberOfIds;
} lIDs;

layout(binding=1, set=2) uniform sampler2DShadow shadowMaps[64];
layout(binding=2, set=2) uniform SHADOW_MAP_MATRICES {
	mat4[64] matrices;
} shadowMapMatrices;

layout(location = 0) out vec4 colorOut;
layout(location = 1) out vec4 resolve;

struct Rng {
    uvec2 seed;
};

float uintToFloat(uint x) {
    uint a = 0x3f800000;
    uint u = a | (x >> 9);

    return uintBitsToFloat(u) - 1.f;
}

void rngInit(out Rng rng, uint i, uint j, uint offset, uint seed){
    uvec2 state = uvec2(i, j) ^ uvec2(offset << 16, (seed+offset) << 16);
    rng.seed = state;
}

// returns random float between (0,1]
float rngRndFloat(inout Rng rng) {
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
vec2 rngRndVec2(inout Rng rng) {
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
vec3 rngRndVec3(inout Rng rng) {
    return vec3(rngRndFloat(rng), rngRndVec2(rng));
}

float aotests(vec3 positionIn) {
	Rng rng;
	rngInit(rng, 200, 100, 82394, 2198);
	float OFFSETS = 0.1f;
	uint passes = 0;
	uint SAMPLE_AMOUNT = 64;
	uint sampleAmount = SAMPLE_AMOUNT;
	for(uint i = 0; i < SAMPLE_AMOUNT; i++) {
		vec3 offset = (rngRndVec3(rng) - vec3(0.5f)) * OFFSETS;
		vec3 posToCheck = positionIn + offset;
		vec4 newProj = camera.mvp * vec4(posToCheck, 1.0f);
		newProj /= newProj.w;
		vec2 uvNeighbor = (newProj.xy + vec2(1.0f)) / 2.0f;
		if(uvNeighbor.x < 0.0f || uvNeighbor.y < 0.0f
			|| uvNeighbor.x > 1.0f || uvNeighbor.y > 1.0f) {
			sampleAmount--;
			continue;
		}
		float depthValue = texture(DEPTHINPUT, uvNeighbor).r;
		if(depthValue >= newProj.z) {
			passes++;
		}
	}
	return min(1.0f, 2 * float(passes) / float(SAMPLE_AMOUNT));
}

struct RenderTillingInfo {
    uint xAmount;
    uint yAmount;
    uint zAmount;

    uint xOffset;
    uint yOffset;
    uint zOffset;
};

layout(push_constant) uniform PC {
    RenderTillingInfo info;
} pc;

void main() {
	float depth = subpassLoad(DEPTH).r;
	const float stepZ = 1/float(pc.info.zAmount);
	const float minZ = stepZ * pc.info.zOffset;
	const float maxZ = minZ + stepZ;
	if(depth > maxZ || depth < minZ) discard;
	uint matID = packUnorm4x8(vec4(subpassLoad(NORMAL).a, 0, 0, 0));
	if(matID == 0) discard;
	vec4 coordsToBackCalc = vec4((2 * (gl_FragCoord.xy / camera.windowSize.xy) - vec2(1.0f)), 0, 1);
	coordsToBackCalc.z = depth;
	coordsToBackCalc /= coordsToBackCalc.w;
	vec4 oldPos = camera.inversVP * coordsToBackCalc;
	oldPos /= oldPos.w;
    vec3 colorIn = subpassLoad(ALBEDO).rgb;
	vec3 colorLight = vec3(0.05);
    vec3 normalIn = normalize(subpassLoad(NORMAL).rgb);

	for(uint i = 0; i < lIDs.numberOfIds; i++) {
		uvec4 packed = lIDs.ids[i >> 2];
		uint li = packed[i & 3];
		LightSource light = lightSources.lights[li];

		// Filter by shadow map if necessary
		if(li < lightSources.numberOfLightsWithShadowMaps) {
			vec4 coords = shadowMapMatrices.matrices[li] * oldPos;
			coords /= coords.w;
			vec3 toCompare = vec3((coords.xy + vec2(1.0f)) / vec2(2.0f), coords.z - 0.0001);
			if(texture(shadowMaps[li], toCompare) != 1.0) {
				continue;
			}

			// Keep this discard check after the texture lookup to avoid UB due to non-dynamically-uniform texture accesses
			if(toCompare.x < 0.0 || toCompare.x > 1.0 || toCompare.y < 0.0 || toCompare.y > 1.0) {
				continue;
			}
		}

		vec3 dist = normalize(light.position - oldPos.xyz);
		float distance = length(light.position - oldPos.xyz);

		colorLight += light.color * light.brightness * max(0, dot(normalIn, dist)) / (distance  * distance * 0.01);
	}
	vec4 cColor = vec4(colorLight * colorIn, 1);
	if(camera.windowSize.z != 0.0f) {
		cColor.rgb *= aotests(oldPos.xyz);
	}
    colorOut = cColor;
	resolve = cColor;
}