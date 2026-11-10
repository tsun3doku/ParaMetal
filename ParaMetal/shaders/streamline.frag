#version 450

layout(location = 0) in float inSpeed;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 1) uniform sampler2D infernoLut;
layout(set = 0, binding = 2) uniform sampler2D viridisLut;
layout(set = 0, binding = 3) uniform sampler2D inferno2Lut;
layout(set = 0, binding = 4) uniform sampler2D parulaLut;

layout(push_constant) uniform PushConstants {
    float minSpeed;
    float maxSpeed;
    uint palette;
    float padding;
} push;

vec3 srgbToLinear(vec3 c) {
    vec3 clamped = max(c, vec3(0.0));
    vec3 low = clamped / 12.92;
    vec3 high = pow((clamped + 0.055) / 1.055, vec3(2.4));
    bvec3 useLow = lessThanEqual(clamped, vec3(0.04045));
    return vec3(
        useLow.x ? low.x : high.x,
        useLow.y ? low.y : high.y,
        useLow.z ? low.z : high.z);
}

vec3 samplePaletteLut(sampler2D lut, float normalized) {
    const float paletteSteps = 32.0;
    float scaled = clamp(normalized, 0.0, 1.0) * (paletteSteps - 1.0);
    float lowerStep = floor(scaled);
    float upperStep = min(lowerStep + 1.0, paletteSteps - 1.0);
    float interpolation = fract(scaled);

    ivec2 lutSize = textureSize(lut, 0);
    int sampleX = lutSize.x / 2;
    int lowerY = int(round((lowerStep / (paletteSteps - 1.0)) * float(lutSize.y - 1)));
    int upperY = int(round((upperStep / (paletteSteps - 1.0)) * float(lutSize.y - 1)));

    vec3 lowerSrgb = texelFetch(lut, ivec2(sampleX, lowerY), 0).rgb;
    vec3 upperSrgb = texelFetch(lut, ivec2(sampleX, upperY), 0).rgb;
    return srgbToLinear(mix(lowerSrgb, upperSrgb, interpolation));
}

void main() {
    float speedRange = max(push.maxSpeed - push.minSpeed, 1e-6);
    float normalizedSpeed = (inSpeed - push.minSpeed) / speedRange;
    float paletteCoordinate = 1.0 - normalizedSpeed;

    vec3 color;
    if (push.palette == 0u) {
        color = samplePaletteLut(inferno2Lut, paletteCoordinate);
    } else if (push.palette == 1u) {
        color = samplePaletteLut(parulaLut, paletteCoordinate);
    } else if (push.palette == 2u) {
        color = samplePaletteLut(viridisLut, paletteCoordinate);
    } else {
        color = samplePaletteLut(infernoLut, paletteCoordinate);
    }

    outColor = vec4(color, 1.0);
}
