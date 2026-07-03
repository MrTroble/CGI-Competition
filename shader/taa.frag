#version 460

layout(binding=0, set=1) uniform CAMERA {
	mat4 mvp;
	mat4 oldViewMatrix;
	mat4 inversMV;
	vec4 windowSize;
} camera;

layout(location = 0) out vec4 colorOut;

layout(binding=0, set=0) uniform sampler2D renderOld;
layout(binding=1, set=0) uniform sampler2D depth;
layout(binding=2, set=0) uniform sampler2D normalTexOld;
layout(binding=3, set=0) uniform sampler2D normalTexNew;
layout(binding=4, set=0) uniform sampler2D resolveNew;

vec3 clipColorTo(vec3 color) {
	vec3 minColor = vec3(0, 0, 0);
	vec3 maxColor = vec3(1.0f, 1.0f, 1.0f);
	for(int i = -1; i < 2; i++) {
		for(int j = -1; j < 2; j++) {
			vec2 coords = gl_FragCoord.xy + ivec2(i, j);
			vec2 uvNew = coords / camera.windowSize.xy;
			if(uvNew.x < 0.0f || uvNew.y < 0.0f) continue;
			if(uvNew.x > 1.0f || uvNew.y > 1.0f) continue;
			vec3 color = texture(resolveNew, uvNew).rgb;
			minColor = max(minColor, color);
			maxColor = min(maxColor, color);
		}
	}
	return max(min(minColor, color), maxColor);
}

void main() {
	vec2 uvNew = gl_FragCoord.xy / camera.windowSize.xy;
	vec4 coordsToBackCalc = vec4((2 * uvNew - vec2(1.0f)), 0, 1);
	coordsToBackCalc.z = texture(depth, uvNew).r;
	vec4 oldPosOut = camera.oldViewMatrix * (camera.inversMV * coordsToBackCalc);
	vec2 uvOld = ((oldPosOut.xy / oldPosOut.w) + vec2(1.0f)) / 2.0f;
	if(uvOld.x < 0 || uvOld.y < 0 || uvOld.x > 1.0f || uvOld.y > 1.0f) discard;
	vec3 cOut = texture(renderOld, uvOld).rgb;
	colorOut = vec4(clipColorTo(cOut), 0.5);
}