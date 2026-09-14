#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(push_constant) uniform PushConstants {
    vec4 corners[8];
} pc;

layout(location = 0) out vec3 fragColor;

const int edgeCorners[24] = int[](
    0, 1,
    0, 2,
    0, 4,

    1, 3,
    1, 5,

    2, 3,
    2, 6,

    3, 7,

    4, 5,
    4, 6,

    5, 7,

    6, 7
);

void main() {
    vec3 p = pc.corners[edgeCorners[gl_VertexIndex]].xyz;
    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
    fragColor = vec3(1.0, 0.55, 0.1);
}
