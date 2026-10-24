#pragma once

#include "HeatBoundaryConditions.hpp"

#include <cstdint>
#include <string>

enum class HeatMaterialPresetId : uint8_t {
    Aluminum = 0,
    Copper = 1,
    Iron = 2,
    Ceramic = 3,
    Custom = 4,
};

struct HeatMaterialPreset {
    HeatMaterialPresetId id = HeatMaterialPresetId::Aluminum;
    const char* name = "Aluminum";
    float density = 2700.0f;
    float specificHeat = 900.0f;
    float conductivity = 205.0f;
};

const HeatMaterialPreset& heatMaterialPresetById(HeatMaterialPresetId id);
const char* heatMaterialPresetName(HeatMaterialPresetId id);

struct HeatSimDefaults {
    static constexpr float ambientTemperatureC = 5.0f;

    static constexpr float density = 1000.0f;
    static constexpr float specificHeat = 1000.0f;
    static constexpr float conductivity = 1.0f;

    static constexpr float fluidDensity = 1.225f;
    static constexpr float fluidSpecificHeat = 1005.0f;
    static constexpr float fluidConductivity = 0.026f;
    static constexpr float fluidKinematicViscosity = 1.516e-5f;

    static constexpr float minNormalDot = -0.65f;
    static constexpr float contactRadius = 0.01f;

    // Interfacial contact conductance per unit area (W/(m^2 K)) between coupled solids.
    static constexpr float contactConductance = 1.0e4f;
};
