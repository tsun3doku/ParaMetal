#version 450

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(push_constant) uniform PushConstants {
    vec4 boxMin;
    vec4 boxMax;
    vec4 selectedColor;
    vec4 hoverColor;
    int selectedFace;
    int hoverFace;
} pc;

layout(location = 0) out vec3 fragColor;
layout(location = 1) out float fragAlpha;

vec3 cornerPos(int corner) {
    return vec3(((corner & 1) != 0) ? pc.boxMax.x : pc.boxMin.x,
                ((corner & 2) != 0) ? pc.boxMax.y : pc.boxMin.y,
                ((corner & 4) != 0) ? pc.boxMax.z : pc.boxMin.z);
}

void main() {
    const bool hovered = gl_VertexIndex >= 6;
    const int face = (hovered && pc.hoverFace == pc.selectedFace) ? -1 : (hovered ? pc.hoverFace : pc.selectedFace);
    const int axis = face >> 1;
    const int lower = face & 1;
    const int axisB = (axis + 1) % 3;
    const int axisC = (axis + 2) % 3;

    const int quadOrder[6] = int[](0, 1, 3, 0, 3, 2);
    const int quadCorner = quadOrder[gl_VertexIndex % 6];

    int corner = lower << axis;
    if ((quadCorner & 1) != 0) corner |= 1 << axisB;
    if ((quadCorner & 2) != 0) corner |= 1 << axisC;

    gl_Position = face < 0 ? vec4(0.0, 0.0, -1.0, 1.0) : ubo.proj * ubo.view * vec4(cornerPos(corner), 1.0);
    fragColor = hovered ? pc.hoverColor.rgb : pc.selectedColor.rgb;
    fragAlpha = hovered ? pc.hoverColor.a : pc.selectedColor.a;
}
