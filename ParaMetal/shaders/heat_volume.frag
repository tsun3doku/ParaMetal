#version 450

layout(location = 0) in vec2 inUV;
layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUBO {
    mat4 model;
    mat4 view;
    mat4 proj;
    vec3 color;
} camera;

layout(set = 0, binding = 1) uniform sampler3D fluidTemperatureVolume;
layout(set = 0, binding = 2) uniform sampler2D infernoLut;
layout(set = 0, binding = 3) uniform sampler2D viridisLut;
layout(set = 0, binding = 4) uniform sampler2D inferno2Lut;
layout(set = 0, binding = 5) uniform sampler2D parulaLut;

layout(push_constant) uniform PushConstants {
    mat4 worldToDomain;
    vec3 gridMin;
    float cellSize;
    ivec3 gridDim;
    float opacity;
    float minTemperature;
    float maxTemperature;
    uint palette;
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

bool intersectBox(vec3 ro, vec3 rd, vec3 boxMin, vec3 boxMax, out float tNear, out float tFar) {
    vec3 invD = 1.0 / rd;
    vec3 t0 = (boxMin - ro) * invD;
    vec3 t1 = (boxMax - ro) * invD;
    vec3 tMin = min(t0, t1);
    vec3 tMax = max(t0, t1);
    tNear = max(max(tMin.x, tMin.y), tMin.z);
    tFar = min(min(tMax.x, tMax.y), tMax.z);
    return tNear <= tFar && tFar > 0.0;
}

void main() {
    if (push.cellSize <= 0.0 || push.gridDim.x <= 0 || push.gridDim.y <= 0 || push.gridDim.z <= 0) {
        outColor = vec4(0.0);
        return;
    }

    mat4 invView = inverse(camera.view);
    mat4 invProj = inverse(camera.proj);

    vec4 clipNear = vec4(inUV * 2.0 - 1.0, 0.0, 1.0);
    vec4 viewNear = invProj * clipNear;
    viewNear /= viewNear.w;
    vec4 worldNear = invView * viewNear;

    vec4 clipFar = vec4(inUV * 2.0 - 1.0, 1.0, 1.0);
    vec4 viewFar = invProj * clipFar;
    viewFar /= viewFar.w;
    vec4 worldFar = invView * viewFar;

    vec3 ro = invView[3].xyz;
    vec3 rd = normalize(worldFar.xyz - worldNear.xyz);

    vec3 domainRo = (push.worldToDomain * vec4(ro, 1.0)).xyz;
    vec3 domainRd = (push.worldToDomain * vec4(rd, 0.0)).xyz;

    float tNear, tFar;
    if (!intersectBox(domainRo, domainRd, vec3(0.0), vec3(1.0), tNear, tFar)) {
        outColor = vec4(0.0);
        return;
    }

    tNear = max(tNear, 0.0);
    if (tNear >= tFar) {
        outColor = vec4(0.0);
        return;
    }

    float dt = push.cellSize * 0.5;
    float t = tNear + 0.5 * dt;
    vec4 accum = vec4(0.0);

    int maxSteps = min(int((tFar - tNear) / dt) + 1, 512);

    for (int i = 0; i < maxSteps; ++i) {
        if (t > tFar) break;

        vec3 pos = ro + rd * t;
        vec3 uvw = ((pos - push.gridMin) / push.cellSize + vec3(0.5)) / vec3(push.gridDim);

        if (all(greaterThanEqual(uvw, vec3(0.0))) && all(lessThanEqual(uvw, vec3(1.0)))) {
            float temp = texture(fluidTemperatureVolume, uvw).r;

            // Solid heatsink: immediate termination
            if (temp < -1.5e30) {
                break;
            }

            // Valid fluid temperature
            if (temp > -0.5e30) {
                vec3 col = heatColorFromTemperature(temp);
                float alphaStep = 1.0 - exp(-push.opacity * 15.0 * dt);

                accum.rgb += (1.0 - accum.a) * col * alphaStep;
                accum.a += (1.0 - accum.a) * alphaStep;

                if (accum.a >= 0.99) {
                    break;
                }
            }
        }

        t += dt;
    }

    outColor = accum;
}
