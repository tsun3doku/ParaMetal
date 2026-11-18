float3 srgbToLinear(float3 c) {
    float3 clamped = max(c, float3(0.0f));
    float3 low = clamped / 12.92f;
    float3 high = pow((clamped + 0.055f) / 1.055f, float3(2.4f));
    bool3 useLow = clamped <= float3(0.04045f);
    return float3(
        useLow.x ? low.x : high.x,
        useLow.y ? low.y : high.y,
        useLow.z ? low.z : high.z);
}

float3 samplePaletteLut(Sampler2D lut, float normalized) {
    const float paletteSteps = 32.0f;
    float scaled = clamp(normalized, 0.0f, 1.0f) * (paletteSteps - 1.0f);
    float lowerStep = floor(scaled);
    float upperStep = min(lowerStep + 1.0f, paletteSteps - 1.0f);
    float interpolation = fract(scaled);

    uint lutWidth = 0u;
    uint lutHeight = 0u;
    lut.GetDimensions(lutWidth, lutHeight);
    int sampleX = int(lutWidth) / 2;
    int lowerY = int(round((lowerStep / (paletteSteps - 1.0f)) * float(int(lutHeight) - 1)));
    int upperY = int(round((upperStep / (paletteSteps - 1.0f)) * float(int(lutHeight) - 1)));

    float3 lowerSrgb = lut.Load(int3(sampleX, lowerY, 0)).rgb;
    float3 upperSrgb = lut.Load(int3(sampleX, upperY, 0)).rgb;
    return srgbToLinear(lerp(lowerSrgb, upperSrgb, interpolation));
}

float3 velocityColor(
    Sampler2D infernoLut,
    Sampler2D viridisLut,
    Sampler2D inferno2Lut,
    Sampler2D parulaLut,
    float speed,
    uint palette,
    float minSpeed,
    float maxSpeed)
{
    float speedRange = max(maxSpeed - minSpeed, 1e-6f);
    float normalizedSpeed = (speed - minSpeed) / speedRange;
    float paletteCoordinate = 1.0f - normalizedSpeed;

    if (palette == 0u) {
        return samplePaletteLut(inferno2Lut, paletteCoordinate);
    } else if (palette == 1u) {
        return samplePaletteLut(parulaLut, paletteCoordinate);
    } else if (palette == 2u) {
        return samplePaletteLut(viridisLut, paletteCoordinate);
    }
    return samplePaletteLut(infernoLut, paletteCoordinate);
}
