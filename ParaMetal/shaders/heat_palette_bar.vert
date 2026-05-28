#version 450

layout(push_constant) uniform PushConstants {
    vec4 barRect;
    vec2 viewportSize;
    float minTemp;
    float maxTemp;
} push;

layout(location = 0) out vec2 outBarUV;

const vec2 positions[4] = vec2[](
    vec2(-0.5, -0.5),
    vec2( 0.5, -0.5),
    vec2( 0.5,  0.5),
    vec2(-0.5,  0.5)
);

const int indices[6] = int[](0, 1, 2, 0, 2, 3);

void main() {
    int idx = indices[gl_VertexIndex];
    vec2 pos = positions[idx];

    outBarUV = pos + 0.5;

    vec2 pixelPos = vec2(
        push.barRect.x + (pos.x + 0.5) * push.barRect.z,
        push.barRect.y + (pos.y + 0.5) * push.barRect.w
    );

    vec2 ndcPos;
    ndcPos.x = (pixelPos.x / push.viewportSize.x) * 2.0 - 1.0;
    ndcPos.y = 1.0 - ((pixelPos.y / push.viewportSize.y) * 2.0);

    gl_Position = vec4(ndcPos, 0.0, 1.0);
}
