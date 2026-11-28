#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

layout(location = 4) in vec3 fragBaryCoord;

layout(location = 0) out uint outPickId;

layout(set = 0, binding = 1) uniform isamplerBuffer S;
layout(set = 0, binding = 2) uniform samplerBuffer A;
layout(set = 0, binding = 3) uniform isamplerBuffer H;
layout(set = 0, binding = 4) uniform isamplerBuffer E;
layout(set = 0, binding = 5) uniform isamplerBuffer T;
layout(set = 0, binding = 6) uniform samplerBuffer L;
layout(set = 0, binding = 7) uniform isamplerBuffer H_input;
layout(set = 0, binding = 9) uniform isamplerBuffer T_input;
layout(set = 0, binding = 10) uniform samplerBuffer L_input;
layout(set = 0, binding = 11) uniform sampler2D wireframe;
layout(set = 0, binding = 12) uniform usamplerBuffer faceIds;

#include "intrinsic_walk.glsl"

void main() {
    ivec3 faceHEs;
    vec2 triCoords[3];
    if (!loadInputTriangleChart(gl_PrimitiveID, faceHEs, triCoords)) {
        outPickId = 0u;
        return;
    }

    vec2 p = inputChartPoint(triCoords, fragBaryCoord);
    vec2 px = p + dFdx(p);
    vec2 py = p + dFdy(p);
    WalkResult walk = locateIntrinsicTriangle(faceHEs, triCoords, p, px, py, false);
    if (walk.status != WALK_SUCCESS) {
        outPickId = 0u;
        return;
    }

    outPickId = texelFetch(faceIds, walk.triangle).r + 1u;
}
