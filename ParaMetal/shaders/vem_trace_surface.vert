#version 450

layout(location = 0) in vec3 inPositionModel;
layout(location = 1) in vec3 inNormalModel;
layout(location = 2) in uint inTraceIndex;

layout(location = 0) out vec3 fragPosition;
layout(location = 1) flat out uint fragTraceIndex;

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    vec4 sourceParams;
    uint palette;
    float minTemperature;
    float maxTemperature;
    float isocontourSpacing;
    uint traceCount;
    uint padding0;
    uint padding1;
    uint padding2;
} push;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

void main() {
    vec4 worldPosition = push.modelMatrix * vec4(inPositionModel, 1.0);

    fragPosition = worldPosition.xyz;
    fragTraceIndex = inTraceIndex;

    gl_Position = ubo.proj * ubo.view * worldPosition;
}
