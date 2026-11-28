#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

layout(location = 4) in vec3 fragBaryCoord;

layout(location = 0) out vec4 outColor;

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

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    vec4 sourceParams;
    vec4 selectionParams;
    vec4 wireframeColor;
} push;

#include "intrinsic_walk.glsl"

void main() {
    int inputTri = gl_PrimitiveID;

    ivec3 faceHEs;
    vec2 triCoords[3];
    if (!loadInputTriangleChart(inputTri, faceHEs, triCoords)) {
        outColor = vec4(0.0, 1.0, 1.0, 1.0);
        return;
    }

    vec2 p = inputChartPoint(triCoords, fragBaryCoord);
    vec2 px = p + dFdx(p);
    vec2 py = p + dFdy(p);
    WalkResult walk = locateIntrinsicTriangle(faceHEs, triCoords, p, px, py, true);

    if (walk.status != WALK_SUCCESS) {
        outColor = vec4(0.8, 0.8, 0.8, 1.0);
        return;
    }

    vec3 dx = walk.baryDx - walk.bary;
    vec3 dy = walk.baryDy - walk.bary;

    float x = textureGrad(
        wireframe,
        vec2(walk.bary.x, 0.5),
        vec2(dx.x, 0.0),
        vec2(dy.x, 0.0)).a;
    float y = textureGrad(
        wireframe,
        vec2(walk.bary.y, 0.5),
        vec2(dx.y, 0.0),
        vec2(dy.y, 0.0)).a;
    float z = textureGrad(
        wireframe,
        vec2(walk.bary.z, 0.5),
        vec2(dx.z, 0.0),
        vec2(dy.z, 0.0)).a;

    float coverage = max(x, max(y, z));
    vec3 lineColor = push.wireframeColor.rgb;
    outColor = vec4(lineColor * coverage, coverage);
}
