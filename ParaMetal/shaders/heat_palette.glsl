#ifndef HEAT_PALETTE_GLSL
#define HEAT_PALETTE_GLSL

const float TEMPERATURE_SCALE = 100.0;

vec3 temperatureToColor(float t) {
    if (t <= 0.0) return vec3(0, 0, 0);
    else if (t < 0.25) return mix(vec3(0, 0, 0), vec3(0.1, 0.0, 0.6), t * 4.0);
    else if (t < 0.375) return mix(vec3(0.1, 0.0, 0.6), vec3(0.3, 0.0, 0.5), (t - 0.25) * 8.0);
    else if (t < 0.55) return mix(vec3(0.3, 0.0, 0.5), vec3(0.9, 0.0, 0.0), (t - 0.375) * 5.71);
    else if (t < 0.75) return mix(vec3(0.9, 0.0, 0.0), vec3(0.9, 0.6, 0.0), (t - 0.55) * 5.0);
    else if (t < 0.9) return mix(vec3(0.9, 0.6, 0.0), vec3(1.0, 1.0, 0.3), (t - 0.75) * 6.67);
    else return mix(vec3(1.0, 1.0, 0.3), vec3(1.0, 1.0, 1.0), (t - 0.9) * 10.0);
}

#endif
