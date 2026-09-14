#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inNormal;

layout(location = 0) out vec3 fragNormal;
layout(location = 1) out vec3 fragWorldPos;

void main() {
    fragNormal = inNormal;
    fragWorldPos = inPos;
    gl_Position = ubo.proj * ubo.view * vec4(inPos, 1.0);
}
