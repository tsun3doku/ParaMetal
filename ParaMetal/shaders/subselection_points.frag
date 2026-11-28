#version 450

layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;

void main() {
    vec2 offset = gl_PointCoord * 2.0 - 1.0;
    if (dot(offset, offset) > 1.0) {
        discard;
    }
    outColor = inColor;
}
