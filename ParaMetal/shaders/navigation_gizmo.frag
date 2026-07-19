#version 450

layout(location = 0) in vec3 localPosition;
layout(location = 1) flat in vec3 faceNormal;
layout(location = 2) in vec2 faceUv;

layout(push_constant) uniform PushConstants {
    mat4 rotation;
    uint hoveredRegion;
    uint pressedRegion;
} pc;

layout(set = 0, binding = 0) uniform sampler2D labelAtlas;

layout(location = 0) out vec4 outColor;

int faceIndex(vec3 normal) {
    if (normal.z < -0.5) return 0;
    if (normal.z > 0.5) return 1;
    if (normal.x > 0.5) return 2;
    if (normal.x < -0.5) return 3;
    if (normal.y > 0.5) return 4;
    return 5;
}

uint encodeRegion(ivec3 side) {
    return uint(1 + (side.x + 1) + 3 * (side.y + 1) + 9 * (side.z + 1));
}

void main() {
    ivec3 side = ivec3(round(faceNormal));
    for (int axis = 0; axis < 3; ++axis) {
        if (side[axis] == 0 && abs(localPosition[axis]) >= 0.58) {
            side[axis] = localPosition[axis] >= 0.0 ? 1 : -1;
        }
    }

    uint region = encodeRegion(side);
    vec3 viewNormal = normalize((pc.rotation * vec4(faceNormal, 0.0)).xyz);
    float light = 0.12 + 0.055 * max(dot(viewNormal, normalize(vec3(-0.45, -0.70, 0.55))), 0.0);
    vec3 color = vec3(light, light + 0.005, light + 0.015);
    if (region == pc.hoveredRegion) {
        color = vec3(0.22, 0.23, 0.25);
    }
    if (region == pc.pressedRegion) {
        color = vec3(0.29, 0.30, 0.32);
    }

    float outerEdge = min(
        min(faceUv.x, 1.0 - faceUv.x),
        min(faceUv.y, 1.0 - faceUv.y));
    if (outerEdge < 0.018) {
        color *= 0.48;
    }

    int face = faceIndex(faceNormal);
    vec2 tileSize = vec2(1.0 / 3.0, 1.0 / 2.0);
    vec2 tile = vec2(float(face % 3), float(face / 3));
    vec2 localUv = vec2(1.0 - faceUv.x, 1.0 - faceUv.y);
    vec2 atlasUv = (tile + localUv) * tileSize;
    vec4 label = texture(labelAtlas, atlasUv);
    color = mix(color, label.rgb, label.a);

    outColor = vec4(color, 0.96);
}
