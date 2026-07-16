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

mat3 frame(const vec3 n) {
    const vec3 nz = normalize(n);
    float sign1 = sign(nz.z);
    if(sign1 == 0) sign1 = 1.0f;
    const float a = -1.0f / (sign1 + nz.z);
    const float b = nz.x * nz.y * a;
    const vec3 nx = vec3(1.0f + sign1 * nz.x * nz.x * a, sign1 * b, -sign1 * nz.x);
    const vec3 ny = vec3(b, sign1 + nz.y * nz.y * a, -nz.y);
    return mat3(normalize(nx), normalize(ny), nz);
}

vec3 from(vec2 angles, vec3 n) {
    return frame(n) * vec3(cos(angles.x) * cos(angles.y), sin(angles.x) * cos(angles.y), sin(angles.y));
}

vec3 getDirection(vec2 rand, vec3 n) {
    float e1 = 2 * M_PI * rand.x;
    float e2 = acos(1 - rand.y);
    return from(vec2(e1, e2), n);
}

float GeoTerm(float alpha, vec3 o, vec3 v, vec3 n, vec3 h) {
    if(dot(v, h) <= 0.0f) return 0.0f; 
    if(dot(o, h) <= 0.0f) return 0.0f; 
    float nDotV = dot(v, n);
    float nDotO = dot(o, n);
    float alphaSqrd = alpha * alpha;
    float G1 = nDotO + sqrt((1.0 - alphaSqrd) * nDotV * nDotV + alphaSqrd);
    float G2 = nDotV + sqrt((1.0 - alphaSqrd) * nDotO * nDotO + alphaSqrd);
    if(G1 * G2 <= 0.0f) return 0.0f;
    return 1.0f / (G1 * G2);
}

// GLTF Specs
vec3 conductor_fresnel(vec3 f0, vec3 bsdf, float VdotH) {
  return bsdf * (f0 + (1 - f0) * pow(1 - abs(VdotH), 5));
}

vec3 fresnel_mix(float ior, vec3 base, vec3 layer, float VdotH) {
  float f0 = ((1-ior)/(1+ior));
  f0 *= f0;
  float fr = f0 + (1 - f0)* pow(1 - abs(VdotH), 5);
  return mix(base, layer, fr);
}

vec3 getGGX(vec3 wo, vec3 normalCur, vec2 roughMetal, vec3 color) {
    float alpha = roughMetal.y*roughMetal.y;
    vec3 h = normalize(-gl_WorldRayDirectionEXT + wo);

    float vDotH = dot(-gl_WorldRayDirectionEXT, h);
    float nDotH = dot(normalCur, h);
    float tanThe = tan(acos(nDotH));
    float alphaRoughnessSq = alpha * alpha;
    float f = nDotH * nDotH * (alphaRoughnessSq - 1.0) + 1.0;
    float ggxD = float(alpha * alpha) / float(M_PI * f * f);
    if(f <= 0.0f || nDotH <= 0.0f) ggxD = 0.0f;

    //GGX G
    float ggxG = GeoTerm(alpha, wo, -gl_WorldRayDirectionEXT, normalCur, h);
    vec3 dialectric = fresnel_mix(1.5f, (color / M_PI), vec3(ggxG * ggxD), vDotH);
    vec3 metallic = conductor_fresnel(color, vec3(ggxG * ggxD), vDotH);
    return mix(dialectric, metallic, roughMetal.x);
}

void main()
{
    const InstanceOffsets off = I.offsets[gl_InstanceID];
    const GeometryData geo = G.geometry[gl_GeometryIndexEXT + off.geometryOffset];
    if(geo.materialID < 0) {
       payload.contribution = vec3(0.0, 0.0, 1.0);
       return;
    }
    const MaterialData mat = M.material[geo.materialID + off.materialOffset];
    vec4 color = vec4(0.5f, 0.5f, 0.5f, 1.0f);
    const uint index = gl_PrimitiveID * 3 + geo.offsetIndexLocal + off.primitiveOffset;
    const uvec3 currentIds = uvec3(ID.ids[index], ID.ids[index + 1], ID.ids[index + 2]) + uvec3(geo.offsetVertexLocal);
    const vec3 bary = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec2 uvCur = bary.x * UV.uvs[currentIds.x] + bary.y * UV.uvs[currentIds.y] + bary.z * UV.uvs[currentIds.z];
    if(mat.texID >= 0) {
        color = texture(textures[nonuniformEXT(mat.texID + off.textureOffset)], uvCur);
    }
    vec2 roughMetal = vec2(mat.metallic, mat.roughness);
    if(mat.roughnessID >= 0) {
        roughMetal *= texture(textures[nonuniformEXT(mat.roughnessID + off.textureOffset)], uvCur).yz;
    }
    const vec3 normalCur = normalize(bary.x * NORM.normal[currentIds.x].xyz + bary.y * NORM.normal[currentIds.y].xyz + bary.z * NORM.normal[currentIds.z].xyz);

    const float lambert = max(0, dot(-gl_WorldRayDirectionEXT, normalCur));
    payload.normal = normalCur;
    vec3 cleanPos = gl_WorldRayDirectionEXT * gl_HitTEXT + gl_WorldRayOriginEXT;
    payload.pos = cleanPos + normalCur * 0.001f;
    payload.dir = normalize(getDirection(rngRndVec2(payload.random), normalCur));
    float pdf = M_PI / dot(payload.dir, normalCur);

    payload.contribution *= pdf;

	//vec3 tangentPre = (dYuv.t * dFdx(positionIn) - dXuv.t * dFdy(positionIn)) /
    //   (dXuv.s * dYuv.t - dYuv.s * dXuv.t);
	//vec3 tangent = normalize(tangentPre - normN * dot(normN, tangentPre));
	//vec3 bitangent = normalize(cross(tangent, normalCur));
	//vec3 normaleVTex = mat3(tangent, bitangent, normalCur) * texture(normalTex, uv1In).rgb;

    // NEE
    uint indexLight = clamp(0, L.numberOfLights - 1, uint(L.numberOfLights * rngRndFloat(payload.random)));
    LightSource lS = L.lights[indexLight];

    //payload.spectrum = vec3(roughMetal, 0.0f);return;

    shadowPayload.isOccluded = true;
    vec3 neeDir = normalize(lS.position - payload.pos);
    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, SHADOW_RAY_OFFSET, 0, 0, payload.pos, 0.0f, neeDir, 10000, 1);
    if(!shadowPayload.isOccluded) {
        float pdf = dot(neeDir, normalCur) / length(lS.position - payload.pos) * L.numberOfLights;
        payload.spectrum += lS.color * payload.contribution * lS.brightness * pdf * getGGX(neeDir, normalCur, roughMetal, color.xyz);
    }
    payload.contribution *= getGGX(payload.dir, normalCur, roughMetal, color.xyz);
}