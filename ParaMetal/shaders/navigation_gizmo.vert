#version 450

layout(location = 0) in vec3 inPosition;
layout(location = 1) in vec3 inNormal;
layout(location = 2) in vec2 inUv;

layout(push_constant) uniform PushConstants {
    mat4 rotation;
    uint hoveredRegion;
    uint pressedRegion;
} pc;

layout(location = 0) out vec3 localPosition;
layout(location = 1) flat out vec3 faceNormal;
layout(location = 2) out vec2 faceUv;

void main() {
    vec3 p = (pc.rotation * vec4(inPosition, 0.0)).xyz;
    gl_Position = vec4(p.x * 0.54, -p.y * 0.54, p.z * 0.08 + 0.5, 1.0);
    localPosition = inPosition;
    faceNormal = inNormal;
    faceUv = inUv;
}
