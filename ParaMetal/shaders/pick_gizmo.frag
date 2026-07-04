#version 450

layout(push_constant) uniform PushConstants {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
    float hovered;
    uint pickId;
} pc;

layout(location = 0) out uint outPickId;

void main() {
    outPickId = pc.pickId;
}
