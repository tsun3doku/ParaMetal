#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 fragPosition;
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

struct IntrinsicVertexData {
    vec3 position;
    uint intrinsicVertexId;
    vec3 normal;
    float padding;
};

layout(std430, set = 0, binding = 14) readonly buffer IntrinsicVertices {
    IntrinsicVertexData vertices[];
};

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
    WalkResult walk = locateIntrinsicTriangle(faceHEs, triCoords, p, p, p, false);
    if (walk.status != WALK_SUCCESS) {
        outPickId = 0u;
        return;
    }

    ivec3 compactVertices;
    if (!intrinsicTriangleVertices(walk.triangle, compactVertices)) {
        outPickId = 0u;
        return;
    }

    uint bestVertexId = 0xffffffffu;
    float bestDistance = 1e30;
    for (int i = 0; i < 3; ++i) {
        int vertexIndex = compactVertices[i];
        if (vertexIndex < 0 || vertexIndex >= int(vertices.length())) {
            continue;
        }

        IntrinsicVertexData vertex = vertices[vertexIndex];
        vec3 vertexWorld =
            (push.modelMatrix * vec4(vertex.position, 1.0)).xyz;
        float distanceWorld = length(vertexWorld - fragPosition);
        if (isnan(distanceWorld) || isinf(distanceWorld) ||
            distanceWorld > push.selectionParams.y) {
            continue;
        }

        if (distanceWorld < bestDistance ||
            (distanceWorld == bestDistance &&
             vertex.intrinsicVertexId < bestVertexId)) {
            bestDistance = distanceWorld;
            bestVertexId = vertex.intrinsicVertexId;
        }
    }

    if (bestVertexId == 0xffffffffu) {
        outPickId = 0u;
        return;
    }
    outPickId = bestVertexId + 1u;
}
