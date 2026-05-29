#pragma once

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
    // Physical Constants
    static constexpr float ambientTemperature = 1.0f;
    
    // Default Material Properties
    static constexpr float density = 1000.0f;       
    static constexpr float specificHeat = 1000.0f;  
    static constexpr float conductivity = 1.0f;   
    
    // Simulation Parameters
    static constexpr float contactThermalConductance = 16000.0f;
    static constexpr float minNormalDot = -0.65f;
    static constexpr float contactRadius = 0.01f;
};
