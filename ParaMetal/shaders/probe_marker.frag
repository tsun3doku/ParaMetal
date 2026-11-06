#version 450

layout(set = 0, binding = 1) uniform sampler2D markerTexture;

layout(push_constant) uniform PushConstants {
    vec4 worldPos;      // xyz = worldPosition, w = scale
    vec4 colorTint;     // rgba tint / selection color
    vec2 viewportSize;  // width, height in pixels
    vec2 markerSize;    // width, height in pixels
    uint isSelected;    // 0 = unselected, 1 = selected
    uint pad0;
    uint pad1;
    uint pad2;
} pc;

layout(location = 0) in vec2 fragTexCoord;

layout(location = 0) out vec4 outColor;

// 24 outer samples on circle radius 1.0, 12 inner samples on radius 0.5
const vec2 sampleOffsets[36] = vec2[](
    // Outer circle (24 samples)
    vec2( 1.0000,  0.0000), vec2( 0.9659,  0.2588), vec2( 0.8660,  0.5000), vec2( 0.7071,  0.7071),
    vec2( 0.5000,  0.8660), vec2( 0.2588,  0.9659), vec2( 0.0000,  1.0000), vec2(-0.2588,  0.9659),
    vec2(-0.5000,  0.8660), vec2(-0.7071,  0.7071), vec2(-0.8660,  0.5000), vec2(-0.9659,  0.2588),
    vec2(-1.0000,  0.0000), vec2(-0.9659, -0.2588), vec2(-0.8660, -0.5000), vec2(-0.7071, -0.7071),
    vec2(-0.5000, -0.8660), vec2(-0.2588, -0.9659), vec2( 0.0000, -1.0000), vec2( 0.2588, -0.9659),
    vec2( 0.5000, -0.8660), vec2( 0.7071, -0.7071), vec2( 0.8660, -0.5000), vec2( 0.9659, -0.2588),
    // Inner circle (12 samples)
    vec2( 0.5000,  0.0000), vec2( 0.4330,  0.2500), vec2( 0.2500,  0.4330), vec2( 0.0000,  0.5000),
    vec2(-0.2500,  0.4330), vec2(-0.4330,  0.2500), vec2(-0.5000,  0.0000), vec2(-0.4330, -0.2500),
    vec2(-0.2500, -0.4330), vec2( 0.0000, -0.5000), vec2( 0.2500, -0.4330), vec2( 0.4330, -0.2500)
);

void main() {
    float aCenter = texture(markerTexture, fragTexCoord).a;

    // Stroke width in physical screen pixels
    float strokePixels = 2.5;
    vec2 stepUV = vec2(strokePixels) / pc.markerSize;

    float maxNeighborA = aCenter;
    for (int i = 0; i < 36; ++i) {
        float sampleA = texture(markerTexture, fragTexCoord + sampleOffsets[i] * stepUV).a;
        maxNeighborA = max(maxNeighborA, sampleA);
    }

    if (maxNeighborA < 0.05) {
        discard;
    }

    // Body color: pure vibrant red
    vec3 bodyColor = vec3(0.92, 0.04, 0.04);

    // Stroke color: thick black when unselected, theme highlight select color (orange #FF9000) when selected
    vec3 highlightOrange = (length(pc.colorTint.rgb) > 0.01) ? pc.colorTint.rgb : vec3(1.0, 0.55, 0.05);
    vec3 blackStroke = vec3(0.04, 0.04, 0.04);
    vec3 strokeColor = (pc.isSelected != 0u) ? highlightOrange : blackStroke;

    // Body coverage factor: transition smoothly from edge to solid interior
    float bodyT = smoothstep(0.2, 0.7, aCenter);

    // Stroke coverage factor: antialiased outer stroke silhouette
    float strokeT = smoothstep(0.08, 0.35, maxNeighborA);

    vec3 finalRgb = mix(strokeColor, bodyColor, bodyT);
    outColor = vec4(finalRgb, strokeT);
}
