#version 450
#extension GL_ARB_separate_shader_objects : enable
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 fragPosition;
layout(location = 1) in vec3 fragNormal;
layout(location = 4) in vec3 fragBaryCoord;

layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gNormal;
layout(location = 2) out vec4 gPosition;
layout(location = 3) out vec4 gMaterial;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(set = 0, binding = 15) uniform MaterialUniformBufferObject {
    vec4 baseColorRoughness;
    vec4 materialParams;
} materialUbo;

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
layout(set = 0, binding = 13) uniform usamplerBuffer selectedElements;

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
    vec4 wireframeColor;
} push;

const float INTRINSIC_ROUGHNESS = 0.05;
const float INTRINSIC_METALNESS = 0.0;
const float INTRINSIC_LIGHTING_MIX = 0.5;

#include "intrinsic_walk.glsl"

vec4 intrinsicMaterial() {
    return vec4(INTRINSIC_ROUGHNESS, INTRINSIC_METALNESS, INTRINSIC_LIGHTING_MIX, 0.0);
}

vec4 unlitMaterial() {
    return vec4(INTRINSIC_ROUGHNESS, INTRINSIC_METALNESS, 0.0, 0.0);
}

vec4 modelMaterial() {
    return vec4(
        clamp(materialUbo.baseColorRoughness.w, 0.04, 1.0),
        clamp(materialUbo.materialParams.x, 0.0, 1.0),
        1.0,
        0.0);
}

vec4 modelUnlitMaterial() {
    return vec4(
        clamp(materialUbo.baseColorRoughness.w, 0.04, 1.0),
        clamp(materialUbo.materialParams.x, 0.0, 1.0),
        0.0,
        0.0);
}

void writeSurface(vec3 albedo, vec4 material) {
    gAlbedo = vec4(albedo, 1.0);
    gNormal = vec4(normalize(fragNormal), 0.0);
    gPosition = vec4(fragPosition, 0.0);
    gMaterial = material;
}

void writeIntrinsicSurface(vec3 albedo) {
    writeSurface(albedo, intrinsicMaterial());
}

vec3 applyWireMask(vec3 baseColor, float wireMask, vec3 edgeColor) {
    return mix(baseColor, edgeColor, wireMask);
}

vec3 intrinsicPaletteColor(int triangleID) {
    const float goldenRatio = 0.618033988749895;
    float randomVal = fract(float(triangleID) * goldenRatio);
    int index = int(randomVal * 7.0);

    vec3 palette[7] = vec3[](
        vec3(0.510, 0.765, 0.941),
        vec3(0.812, 0.922, 1.000),
        vec3(0.976, 0.922, 0.651),
        vec3(1.000, 0.820, 0.871),
        vec3(1.000, 0.600, 0.750),
        vec3(0.647, 0.949, 0.776),
        vec3(0.780, 0.640, 0.940)
    );

    vec3 linearColor = pow(palette[index], vec3(2.2));
    float saturation = 1.2;
    float luminance = dot(linearColor, vec3(0.2126, 0.7152, 0.0722));
    return mix(vec3(luminance), linearColor, saturation);
}

void main() {
    int inputTri = gl_PrimitiveID;

    ivec3 faceHEs;
    vec2 triCoords[3];
    if (!loadInputTriangleChart(inputTri, faceHEs, triCoords)) {
        writeIntrinsicSurface(vec3(0, 1, 1));
        return;
    }

    vec2 p = inputChartPoint(triCoords, fragBaryCoord);
    vec2 px = p + dFdx(p);
    vec2 py = p + dFdy(p);
    WalkResult walk = locateIntrinsicTriangle(faceHEs, triCoords, p, px, py, true);

    if (walk.status != WALK_SUCCESS) {
        writeIntrinsicSurface(vec3(0.8, 0.8, 0.8));
        return;
    }

    bool selected = false;
    if (push.sourceParams.y > 0.5) {
        int selectionType = int(push.selectionParams.x + 0.5);

        if (selectionType == 2) {
            selected = texelFetch(selectedElements, walk.triangle).r != 0u;
        } else if (selectionType == 0) {
            ivec3 compactVertices;
            if (intrinsicTriangleVertices(walk.triangle, compactVertices)) {
                float closest = 1e30;
                for (int i = 0; i < 3; ++i) {
                    int vertexIndex = compactVertices[i];
                    if (vertexIndex < 0 ||
                        vertexIndex >= int(vertices.length()) ||
                        texelFetch(selectedElements, vertexIndex).r == 0u) {
                        continue;
                    }
                    vec3 vertexWorld = (push.modelMatrix * vec4(vertices[vertexIndex].position, 1.0)).xyz;
                    float distanceWorld = length(vertexWorld - fragPosition);
                    if (distanceWorld <= push.selectionParams.y &&
                        distanceWorld < closest) {
                        closest = distanceWorld;
                        selected = true;
                    }
                }
            }
        } else if (selectionType == 1) {
            uvec3 edgeIds;
            if (intrinsicTriangleEdgeIds(walk.triangle, edgeIds)) {
                vec3 distances = intrinsicEdgeDistances(walk);
                float closest = uintBitsToFloat(0x7f800000u);
                for (int i = 0; i < 3; ++i) {
                    if (texelFetch(
                            selectedElements, int(edgeIds[i])).r != 0u) {
                        closest = min(closest, distances[i]);
                    }
                }
                selected = closest <= push.selectionParams.z;
            }
        }
    }

    int mode = int(push.selectionParams.w + 0.5);

    if (mode == 0) {
        vec3 modelAlbedo = materialUbo.baseColorRoughness.rgb * ubo.color;
        if (selected) {
            writeSurface(vec3(1.0, 0.45, 0.08), modelUnlitMaterial());
        } else {
            writeSurface(modelAlbedo, modelMaterial());
        }
        return;
    }

    vec3 albedo = selected
        ? vec3(1.0, 0.45, 0.08)
        : intrinsicPaletteColor(walk.triangle);

    vec3 fx = walk.baryDx - walk.bary;
    vec3 fy = walk.baryDy - walk.bary;

    vec4 xWire = textureGrad(wireframe, vec2(walk.bary.x, 0.5), vec2(fx.x, 0.0), vec2(fy.x, 0.0));
    vec4 yWire = textureGrad(wireframe, vec2(walk.bary.y, 0.5), vec2(fx.y, 0.0), vec2(fy.y, 0.0));
    vec4 zWire = textureGrad(wireframe, vec2(walk.bary.z, 0.5), vec2(fx.z, 0.0), vec2(fy.z, 0.0));

    float edge = max(xWire.a, max(yWire.a, zWire.a));

    albedo = applyWireMask(albedo, edge, push.wireframeColor.rgb);
    if (selected) {
        writeSurface(vec3(1.0, 0.45, 0.08), unlitMaterial());
        return;
    }
    writeIntrinsicSurface(clamp(albedo, vec3(0.0), vec3(1.0)));
}
