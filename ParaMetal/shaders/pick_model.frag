#version 450

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    uint pickId;
    uint elementMode;
    uint primitiveBase;
    uint padding;
} pc;

layout(location = 0) out uint outPickId;

void main() {
    outPickId = (pc.elementMode != 0u)
        ? (uint(gl_PrimitiveID) + pc.primitiveBase + 1u)
        : pc.pickId;
}
