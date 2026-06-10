#version 450
#extension GL_GOOGLE_include_directive : require

layout(location = 0) in vec3 inPosition;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(set = 0, binding = 1, std430) readonly buffer TempBuffer {
    float temperatures[];
} tempBufferB;

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    float pointSize;
    float viewportHeight;
} pc;

layout(location = 0) out vec3 fragColor;

#include "heat_palette.glsl"

void main() {
    vec4 worldPos = pc.modelMatrix * vec4(inPosition, 1.0);
    vec4 viewPos = ubo.view * worldPos;
    gl_Position = ubo.proj * viewPos;

    float distance = gl_Position.w;
    float projScale = abs(ubo.proj[1][1]);
    float screenSize = (pc.pointSize / distance) * projScale * (pc.viewportHeight * 0.5);
    gl_PointSize = clamp(screenSize, 1.0, 64.0);

    float temperature = tempBufferB.temperatures[gl_VertexIndex];
    float normalizedTemp = temperature / TEMPERATURE_SCALE;
    fragColor = temperatureToColor(normalizedTemp);
}
