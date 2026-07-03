#version 460

layout(location = 0) in vec3 positionIn;
layout(location = 1) in vec3 normalIn;
layout(location = 2) in vec2 uv1In;

layout(location = 0) out vec4 albedo;
layout(location = 1) out vec4 normalAndMaterial;

layout(binding=0, set=2) uniform sampler2D albedoTex;
layout(binding=1, set=2) uniform sampler2D normalTex;

layout(push_constant) uniform PC {
	uint matID;
	float alphaCutoff;
} pc;

layout(binding=0, set=1) uniform CAMERA {
	mat4 mvp;
	mat4 normalMVP;
	mat4 inverseVP;
} camera;

layout(binding=0, set=0) uniform MODEL {
	mat4 model;
	mat4 normal;
} m;


void main() {
	vec4 material = unpackUnorm4x8(pc.matID + 1);
	vec3 normN = normalize(normalIn);
	normalAndMaterial = vec4(normN, material.x);
	albedo = vec4(1, 0, 0, 1);
#ifdef USE_ALBEDO
	vec4 colorIn = texture(albedoTex, uv1In);
	if(colorIn.a < pc.alphaCutoff) discard;
	albedo.rgb = colorIn.rgb;
#endif
#ifdef USE_ALB_NORM
	vec4 colorIn = texture(albedoTex, uv1In);
	if(colorIn.a < pc.alphaCutoff) discard;
	albedo.rgb = colorIn.rgb;
	// Tangeten raum TODO
    vec2 dXuv = dFdx(uv1In);
    vec2 dYuv = dFdy(uv1In);
	vec3 tangentPre = (dYuv.t * dFdx(positionIn) - dXuv.t * dFdy(positionIn)) /
        (dXuv.s * dYuv.t - dYuv.s * dXuv.t);
	vec3 tangent = normalize(tangentPre - normN * dot(normN, tangentPre));
	vec3 bitangent = normalize(cross(tangent, normN));
	vec3 normaleVTex = mat3(tangent, bitangent, normN) * texture(normalTex, uv1In).rgb;
	normalAndMaterial.rgb = normaleVTex;
#endif
}