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

vec3 cornerPos(int corner) {
    return vec3(((corner & 1) != 0) ? pc.boxMax.x : pc.boxMin.x,
                ((corner & 2) != 0) ? pc.boxMax.y : pc.boxMin.y,
                ((corner & 4) != 0) ? pc.boxMax.z : pc.boxMin.z);
}

bool onFace(int corner, int face) {
    return ((corner >> (face >> 1)) & 1) == (face & 1);
}

void main() {
    const int line = gl_VertexIndex >> 1;
    const int cornerA = edgeCorners[line * 2];
    const int cornerB = edgeCorners[line * 2 + 1];
    const vec3 p = cornerPos(edgeCorners[gl_VertexIndex]);

    const bool selected = pc.selectedFace >= 0 && onFace(cornerA, pc.selectedFace) && onFace(cornerB, pc.selectedFace);
    const bool hovered = pc.hoverFace >= 0 && onFace(cornerA, pc.hoverFace) && onFace(cornerB, pc.hoverFace);

    gl_Position = ubo.proj * ubo.view * vec4(p, 1.0);
    fragColor = selected ? pc.selectedColor.rgb : (hovered ? pc.hoverColor.rgb : vec3(1.0, 0.55, 0.1));
}
