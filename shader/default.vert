#version 460

layout(location = 0) in vec3 positionIn;
layout(location = 1) in vec3 normalIn;
layout(location = 2) in vec2 uv1In;

layout(location = 0) out vec3 positionOut;
layout(location = 1) out vec3 normalOut;
layout(location = 2) out vec2 uv1Out;

layout(binding=0, set=0) uniform MODEL {
	mat4 model;
	mat4 normalM;
} model;

layout(binding=0, set=1) uniform CAMERA {
	mat4 mvp;
	mat4 normalMVP;
} camera;

out gl_PerVertex {
   vec4 gl_Position;
};

void main() {
	vec4 worldPos = model.model * vec4(positionIn, 1);
	vec4 clipPos = camera.mvp * worldPos;
	gl_Position = clipPos;
	positionOut = worldPos.xyz / worldPos.w;
	normalOut = normalize(model.normalM * vec4(normalIn, 0)).xyz;
	uv1Out = uv1In;
}