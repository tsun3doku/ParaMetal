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

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    vec4 sourceParams;
    vec4 selectionParams;
} push;

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
    WalkResult walk = locateIntrinsicTriangle(faceHEs, triCoords, p, px, py, true);
    if (walk.status != WALK_SUCCESS) {
        outPickId = 0u;
        return;
    }

    uvec3 edgeIds;
    if (!intrinsicTriangleEdgeIds(walk.triangle, edgeIds)) {
        outPickId = 0u;
        return;
    }

    vec3 distances = intrinsicEdgeDistances(walk);
    float bestDistance = distances.x;
    uint bestEdgeId = edgeIds.x;
    if (distances.y < bestDistance ||
        (distances.y == bestDistance && edgeIds.y < bestEdgeId)) {
        bestDistance = distances.y;
        bestEdgeId = edgeIds.y;
    }
    if (distances.z < bestDistance ||
        (distances.z == bestDistance && edgeIds.z < bestEdgeId)) {
        bestDistance = distances.z;
        bestEdgeId = edgeIds.z;
    }

    if (isnan(bestDistance) || isinf(bestDistance) ||
        bestDistance > push.selectionParams.z ||
        bestEdgeId == 0xffffffffu) {
        outPickId = 0u;
        return;
    }

    outPickId = bestEdgeId + 1u;
}
