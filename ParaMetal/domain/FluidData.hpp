#pragma once

// Neutral authored fluid material properties. Zero backend or velocity dependencies.
struct FluidData {
    float density = 1.225f;
    float kinematicViscosity = 1.516e-5f;
    float specificHeat = 1005.0f;
    float conductivity = 0.026f;
    float initialTemperatureC = 20.0f;
};
