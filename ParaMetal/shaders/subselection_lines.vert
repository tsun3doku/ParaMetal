#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    vec4 selectionColor;
    float pointRadiusWorld;
    float viewportHeight;
} pc;

layout(location = 0) in vec3 inPos;
layout(location = 1) in vec3 inColor;
layout(location = 2) in vec3 inNormal;
layout(location = 3) in vec2 inTexCoord;

layout(location = 0) out vec4 outColor;

void main() {
    vec4 viewPosition = ubo.view * pc.modelMatrix * vec4(inPos, 1.0);
    vec4 clipPosition = ubo.proj * viewPosition;
    gl_Position = clipPosition;
    float pointPixels = pc.pointRadiusWorld * abs(ubo.proj[1][1]) *
        pc.viewportHeight / max(abs(clipPosition.w), 1e-5);
    gl_PointSize = clamp(pointPixels, 4.0, 10.0);
    outColor = vec4(inColor, pc.selectionColor.a * inTexCoord.x);
}
