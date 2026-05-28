#version 450
#extension GL_GOOGLE_include_directive : require

#include "heat_palette.glsl"

layout(push_constant) uniform PushConstants {
    vec4 barRect;
    vec2 viewportSize;
    float minTemp;
    float maxTemp;
} push;

layout(location = 0) in vec2 inBarUV;

layout(location = 1) out vec4 outColor;

const float BORDER_THICKNESS_PX = 1.0;

void main() {
    float temp = mix(push.minTemp, push.maxTemp, inBarUV.y);
    float normalized = clamp(temp / push.maxTemp, 0.0, 1.0);
    vec3 color = temperatureToColor(normalized);

    float barHeightPx = push.barRect.w;
    float barWidthPx = push.barRect.z;

    float borderThicknessNormX = BORDER_THICKNESS_PX / barWidthPx;
    float borderThicknessNormY = BORDER_THICKNESS_PX / barHeightPx;

    bool isBorder = inBarUV.x < borderThicknessNormX
                 || inBarUV.x > 1.0 - borderThicknessNormX
                 || inBarUV.y < borderThicknessNormY
                 || inBarUV.y > 1.0 - borderThicknessNormY;

    if (isBorder) {
        color = vec3(0.0, 0.0, 0.0);
    }

    outColor = vec4(color, 1.0);
}
