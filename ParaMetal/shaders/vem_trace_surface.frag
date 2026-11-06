#version 450

layout(location = 0) in vec3 fragPosition;
layout(location = 1) flat in uint fragTraceIndex;

layout(push_constant) uniform PushConstants {
    mat4 modelMatrix;
    vec4 sourceParams;
    uint palette;
    float minTemperature;
    float maxTemperature;
    float isocontourSpacing;
    uint traceCount;
    uint padding0;
    uint padding1;
    uint padding2;
} push;

layout(set = 0, binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} ubo;

layout(set = 0, binding = 1) readonly buffer TraceTemperatures {
    float traceTemperature[];
} traceTemperatures;

layout(set = 0, binding = 2) uniform sampler2D infernoLut;
layout(set = 0, binding = 3) uniform sampler2D viridisLut;
layout(set = 0, binding = 4) uniform sampler2D inferno2Lut;
layout(set = 0, binding = 5) uniform sampler2D parulaLut;
layout(set = 0, binding = 6) uniform sampler2D isocontourTexture;

layout(location = 0) out vec4 gAlbedo;
layout(location = 1) out vec4 gMaterial;

const vec3 traceErrorColor = vec3(1.0, 0.0, 1.0);
const float HEAT_ROUGHNESS = 0.05;
const float HEAT_METALNESS = 0.0;
const float HEAT_LIGHTING_MIX = 0.1;

vec3 srgbToLinear(vec3 c) {
    vec3 clamped = clamp(c, vec3(0.0), vec3(1.0));
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

vec3 heatColorFromTemperature(float temperature) {
    float temperatureRange = max(push.maxTemperature - push.minTemperature, 1e-6);
    float normalizedTemperature = (temperature - push.minTemperature) / temperatureRange;
    float paletteCoordinate = 1.0 - normalizedTemperature;
    if (push.palette == 0u) {
        return samplePaletteLut(inferno2Lut, paletteCoordinate);
    }
    if (push.palette == 1u) {
        return samplePaletteLut(parulaLut, paletteCoordinate);
    }
    if (push.palette == 2u) {
        return samplePaletteLut(viridisLut, paletteCoordinate);
    }
    return samplePaletteLut(infernoLut, paletteCoordinate);
}

vec3 applyIsocontour(vec3 heatColor, float temperature) {
    if (push.isocontourSpacing <= 0.0) return heatColor;
    float u = temperature / push.isocontourSpacing;
    vec2 grad = vec2(dFdx(u), dFdy(u));
    if (length(grad) < 1e-5) return heatColor;
    vec4 lineTex = textureGrad(
        isocontourTexture, vec2(u, 0.5),
        vec2(grad.x, 0.0), vec2(grad.y, 0.0));
    return mix(heatColor, vec3(0.0), lineTex.a);
}

void writeHeatSurface(vec3 albedo) {
    gAlbedo = vec4(albedo, 1.0);
    gMaterial = vec4(HEAT_ROUGHNESS, HEAT_METALNESS, HEAT_LIGHTING_MIX, 0.0);
}

void main() {
    if (fragTraceIndex >= push.traceCount) {
        writeHeatSurface(traceErrorColor);
        return;
    }

    float temperature = traceTemperatures.traceTemperature[fragTraceIndex];
    if (isnan(temperature) || isinf(temperature) || temperature < -1.0e20) {
        writeHeatSurface(traceErrorColor);
        return;
    }

    vec3 heatColor = heatColorFromTemperature(temperature);
    heatColor = applyIsocontour(heatColor, temperature);

    if (push.sourceParams.y > 0.5) {
        heatColor *= ((gl_PrimitiveID & 1) == 0) ? 1.0 : 0.55;
    }

    writeHeatSurface(heatColor);
}
