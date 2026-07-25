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
    if(G1 * G2 == 0.0f) return 0.0f;
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

vec3 getGGX(vec3 wo, vec3 normalCur, vec2 roughMetal, vec3 color, float ior) {
    float alpha = roughMetal.x*roughMetal.x;
    vec3 h = normalize(-gl_WorldRayDirectionEXT + wo);

    float vDotH = dot(-gl_WorldRayDirectionEXT, h);
    float nDotH = dot(normalCur, h);
    float tanThe = tan(acos(nDotH));
    float alphaRoughnessSq = alpha * alpha;
    float f = nDotH * nDotH * (alphaRoughnessSq - 1.0) + 1.0;
    float ggxD = float(alphaRoughnessSq) / float(M_PI * f * f);
    if(nDotH <= 0.0f) ggxD = 0.0f;
    
    //GGX G
    float ggxG = GeoTerm(alpha, wo, -gl_WorldRayDirectionEXT, normalCur, h);
    vec3 dialectric = fresnel_mix(1.5f, (color / M_PI), vec3(ggxG * ggxD), vDotH);
    vec3 metallic = conductor_fresnel(color, vec3(ggxG * ggxD), vDotH);
    return mix(dialectric, metallic, roughMetal.y);
}

vec3 sampleGGX(vec2 rand, vec3 normalCur, float roughness)
{
    float alpha = roughness * roughness;
    float alpha2 = alpha * alpha;


    float phi = 2.0 * M_PI * rand.x;

    float cosTheta = sqrt(
        (1.0 - rand.y) /
        (1.0 + (alpha2 - 1.0) * rand.y)
    );

    float sinTheta = sqrt(max(0.0, 1.0 - cosTheta * cosTheta));

    vec3 hLocal = vec3(
        cos(phi) * sinTheta,
        sin(phi) * sinTheta,
        cosTheta
    );

    return normalize(frame(normalCur) * hLocal);
}


void main()
{
   payload.miss = false;
    const InstanceOffsets off = I.offsets[gl_InstanceID];
    const GeometryData geo = G.geometry[gl_GeometryIndexEXT + off.geometryOffset];
    if(geo.materialID < 0) {
       payload.contribution = vec3(0.0, 0.0, 1.0);
       return;
    }
    const MaterialData mat = M.material[geo.materialID + off.materialOffset];
    vec4 color = vec4(0.5f, 0.5f, 0.5f, 1.0f);
    const uint index = gl_PrimitiveID * 3 + geo.offsetIndexLocal + off.indexOffset;
    const uvec3 currentIds = uvec3(ID.ids[index], ID.ids[index + 1], ID.ids[index + 2]) + uvec3(geo.offsetVertexLocal + off.primitiveOffset);
    const vec3 bary = vec3(1.0f - attribs.x - attribs.y, attribs.x, attribs.y);
    vec2 uvCur = bary.x * UV.uvs[currentIds.x] + bary.y * UV.uvs[currentIds.y] + bary.z * UV.uvs[currentIds.z];
    if(mat.texID >= 0) {
        color = texture(textures[nonuniformEXT(mat.texID + off.textureOffset)], uvCur);
    }
    vec2 roughMetal = vec2(mat.roughness, mat.metallic);
    if(mat.roughnessID >= 0) {
        roughMetal *= texture(textures[nonuniformEXT(mat.roughnessID + off.textureOffset)], uvCur).yz;
    }
    vec3 normalCur = normalize(bary.x * NORM.normal[currentIds.x].xyz + bary.y * NORM.normal[currentIds.y].xyz + bary.z * NORM.normal[currentIds.z].xyz);
    if(dot(normalCur, -gl_WorldRayDirectionEXT) < 0) {
        normalCur *= -1;
    }

    const float lambert = max(0, dot(-gl_WorldRayDirectionEXT, normalCur));
    payload.normal = normalCur;
    vec3 cleanPos = gl_WorldRayDirectionEXT * gl_HitTEXT + gl_WorldRayOriginEXT;
    payload.pos = cleanPos;
    payload.dir = normalize(getDirection(rngRndVec2(payload.random), normalCur));


    float metallic = roughMetal.y;
    float roughness = roughMetal.x;
    vec3 F0;
    if(metallic > 0.0)
    {
       F0 = color.rgb;
    }
    else
    {
        float f0 = pow((mat.ior - 1.0) /(mat.ior + 1.0), 2.0);
        F0 = vec3(f0) * color.rgb;
    }
    float cosTheta = clamp(dot(-gl_WorldRayDirectionEXT, normalCur),0.0,1.0);
    vec3 F = F0 + (1.0 - F0) * pow(1.0 - cosTheta, 5.0);
    float specChance = clamp(max(max(F.r,F.g),F.b),0.05,0.95);
    float r = rngRndFloat(payload.random);

    //-----------------
    // Reflection
    //-----------------
    if(r < specChance){

         //-----------------
         // Perfekte Reflexion
        //-----------------
         if(roughness < 0.001)
         {
        payload.dir = reflect(gl_WorldRayDirectionEXT, normalCur);
        payload.contribution *= F / specChance;
        return;
         }


        //-----------------
        //  Reflexion kaputt Hilfe!
        //-----------------
       vec3 H = sampleGGX(rngRndVec2(payload.random), normalCur, roughness);
       vec3 L = reflect(gl_WorldRayDirectionEXT, H);
       L = normalize(L);
       float NoL = dot(normalCur, L);
       if(NoL <= 0.0)
        {
         payload.contribution = vec3(0.0);
         return;
        }
       float NoV = max(dot(normalCur, -gl_WorldRayDirectionEXT), 0.0001);
       float NoH = max(dot(normalCur, H), 0.0001);
       float VoH = max(dot(-gl_WorldRayDirectionEXT, H), 0.0001);
       float alpha = roughness * roughness;
       float alpha2 = alpha * alpha;
       float denom = NoH * NoH * (alpha2 - 1.0) + 1.0;
       float D = alpha2 / (M_PI * denom * denom);
       float pdf = D * NoH / (4.0 * VoH);
       float G = GeoTerm(alpha, L, -gl_WorldRayDirectionEXT, normalCur, H);
       vec3 brdf = F * D * G;
       //vec3 brdf = getGGX( L,normalCur, roughMetal,color.xyz, mat.ior);
       vec3 weight = brdf * NoL / pdf;

      payload.contribution *= weight / specChance;
      payload.dir = L;
      return;
    }
    else{

//-----------------
// Transmission
//-----------------
    if(mat.transmission > 0.0)
    {
        vec3 N = normalCur;
        float eta;
        if(dot(gl_WorldRayDirectionEXT,N) < 0)
        {
            eta = 1.0 / max(mat.ior,1.01);
        }
        else
        {
            N=-N;
            eta = max(mat.ior,1.01);
        }
        vec3 transmission = refract( gl_WorldRayDirectionEXT,N,eta);
        payload.dir = normalize(transmission);
        payload.contribution *= color.rgb / ((1.0-specChance));
        return;
    }
//-----------------
// NEE
//-----------------
    uint indexLight = min(uint(rngRndFloat(payload.random)*float(L.numberOfLights)), L.numberOfLights-1);
    LightSource lS = L.lights[indexLight];
    vec3 toLight = lS.position - payload.pos;
    float dist = length(toLight);
    vec3 lightDir = toLight/dist;

    shadowPayload.isOccluded = true;
    traceRayEXT(
         topLevelAS,
         gl_RayFlagsOpaqueEXT,
         0xff,
         SHADOW_RAY_OFFSET,
         0,
          0,
          payload.pos,
          0.001,
          lightDir,
          dist-0.001,
          1
        );

    if(!shadowPayload.isOccluded)
    {
    float NdotL = max(dot(normalCur,lightDir),0.0);

    payload.spectrum += payload.contribution *
        getGGX(lightDir, normalCur, roughMetal, color.xyz, mat.ior) *
        lS.color *
        lS.brightness *
        NdotL *
        float(L.numberOfLights) /
        (dist*dist);
    }

    payload.dir = normalize(getDirection(rngRndVec2(payload.random),normalCur));
    float NoL = max(dot(normalCur, payload.dir),0.0);
    payload.contribution *= getGGX(payload.dir, normalCur, roughMetal, color.xyz,mat.ior) / ((1.0-specChance) * 2 * M_PI);
    }
}