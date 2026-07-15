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
       payload.contribution = vec3(1.0, 0.0, 1.0);
       return;
    }
    MaterialData mat = M.material[geo.materialID + off.materialOffset];
    if(mat.texID < 0) {
        payload.contribution = vec3(1.0, 0.0, 0.0);
        return;
    }
    const uint index = gl_PrimitiveID * 3 + geo.offsetIndexLocal + off.primitiveOffset;
    const uvec3 currentIds = uvec3(ID.ids[index], ID.ids[index + 1], ID.ids[index + 2]) + uvec3(geo.offsetVertexLocal);
    const vec3 bary = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    const vec2 uvCur = bary.x * UV.uvs[currentIds.x] + bary.y * UV.uvs[currentIds.y] + bary.z * UV.uvs[currentIds.z];
    vec4 color = texture(textures[nonuniformEXT(mat.texID + off.textureOffset)], uvCur);
    const vec3 normalCur = normalize(bary.x * NORM.normal[currentIds.x].xyz + bary.y * NORM.normal[currentIds.y].xyz + bary.z * NORM.normal[currentIds.z].xyz);

    const float lambert = max(0, dot(-gl_WorldRayDirectionEXT, normalCur));
    payload.contribution *= color.xyz ;
    payload.normal = normalCur;
    vec3 cleanPos = gl_WorldRayDirectionEXT * gl_HitTEXT + gl_WorldRayOriginEXT;
    payload.pos = cleanPos + normalCur * 0.001f;
    payload.dir = normalize(getDirection(rngRndVec2(payload.random)));

	//vec3 tangentPre = (dYuv.t * dFdx(positionIn) - dXuv.t * dFdy(positionIn)) /
    //   (dXuv.s * dYuv.t - dYuv.s * dXuv.t);
	//vec3 tangent = normalize(tangentPre - normN * dot(normN, tangentPre));
	//vec3 bitangent = normalize(cross(tangent, normalCur));
	//vec3 normaleVTex = mat3(tangent, bitangent, normalCur) * texture(normalTex, uv1In).rgb;

    shadowPayload.isOccluded = true;
    uint indexLight = clamp(0, L.numberOfLights - 1, uint(L.numberOfLights * rngRndFloat(payload.random)));
    LightSource lS = L.lights[indexLight];


    //GGX D
    float alpha = mat.roughness * mat.roughness;
    vec3 h = normalize(-gl_WorldRayDirectionEXT + (payload.pos - lS.position ));
    float nDotH = max(0.0, dot(h, normalCur));
    float denom = nDotH * nDotH * (alpha * alpha - 1.0) + 1.0;
    float ggxD = (alpha * alpha) / (M_PI * denom * denom);
    //float ggxD = float(alpha * alpha * normalCur) / float(M_PI * pow(cos(thetaM), 4) * pow(mat.roughness * mat.roughness + pow(tan(thetaM) , 2),2));
  
    //GGX G
    float thetaV = acos(dot(-gl_WorldRayDirectionEXT, payload.dir));
    float tanThetaV = tan(thetaV);
    float ggxG = 2.0 / (1.0 + sqrt(1.0 + alpha * alpha * pow(tanThetaV, 2)));

    vec3 f0 = mix(vec3(0.04), vec3(color), mat.metallic); 
    float vDotH = max(0.0, dot(-gl_WorldRayDirectionEXT, h));
    vec3 ggxF = f0 + (1.0 - f0) * pow(1.0 - vDotH, 5.0);



    float nDotV = max(0.0, dot(normalCur, -gl_WorldRayDirectionEXT));
    float nDotL = max(0.0, dot(normalCur, (payload.pos - lS.position )));
    vec3 specularTerm = (ggxD * ggxF * ggxG) / max(4.0 * nDotV * nDotL, 0.001);

    vec3 kD = (1.0 - ggxF) * (1.0 - mat.metallic);
    vec3 diffuseTerm = kD * color.xyz / M_PI; 

    vec3 bRDF = diffuseTerm + specularTerm;
    payload.spectrum *= bRDF;

    vec3 neeDir = normalize(lS.position - payload.pos);
    traceRayEXT(topLevelAS, gl_RayFlagsOpaqueEXT, 0xff, SHADOW_RAY_OFFSET, 0, 0, payload.pos, 0.01f, neeDir, 10000, 1);
    if(!shadowPayload.isOccluded) {
        float pdf = dot(neeDir, normalCur) / length(lS.position - payload.pos);
        payload.spectrum += lS.color * payload.contribution *  lS.brightness * nDotL;
    }

}