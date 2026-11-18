#version 450

layout(location = 0) in vec4 inPositionSpeed;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(location = 0) out float outSpeed;
layout(location = 1) out float outForwardDepth;

void main() {
    vec4 viewPosition = ubo.view * vec4(inPositionSpeed.xyz, 1.0);
    gl_Position = ubo.proj * viewPosition;
    outSpeed = inPositionSpeed.w;
    outForwardDepth = -viewPosition.z;
}
