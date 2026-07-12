#version 460

out gl_PerVertex {
   vec4 gl_Position;
};

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

const vec2 values[4] = { vec2(0,0), vec2(0,1), vec2(1,1), vec2(1,0) };

void main() {
    vec2 tile = values[gl_VertexIndex];
    vec2 stepForTile = vec2(1 / float(pc.info.xAmount), 1 / float(pc.info.yAmount));
    tile *= stepForTile;
    tile += stepForTile * vec2(pc.info.xOffset, pc.info.yOffset);
    gl_Position = vec4(tile * 2.0f - 1.0f, 0.0f, 1.0f);
}