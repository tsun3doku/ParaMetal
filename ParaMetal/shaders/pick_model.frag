#version 450

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    uint pickId;
} pc;

layout(location = 0) out uint outPickId;

void main() {
    outPickId = pc.pickId;
}
