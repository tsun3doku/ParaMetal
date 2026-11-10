#pragma once

#include <array>
#include <cstdint>

namespace heat {

enum class BoundaryType : uint32_t {
    Adiabatic = 0,
    Dirichlet = 1, // Fixed temperature
    Neumann   = 2, // Fixed heat flux
    Robin     = 3  // Convective (HTC)
};

struct BoundaryCondition {
    BoundaryType type = BoundaryType::Adiabatic;
    float temperatureC = 20.0f;
    float heatFlux = 0.0f;
    float htc = 0.0f;
    float inflowTemperatureC = 20.0f;
};

using DomainBoundaryConditions = std::array<BoundaryCondition, 6>;

inline DomainBoundaryConditions createDefaultDomainBoundaries(float ambientTempC = 20.0f) {
    DomainBoundaryConditions b{};
    for (size_t i = 0; i < 6; ++i) {
        b[i].type = BoundaryType::Adiabatic;
        b[i].temperatureC = ambientTempC;
        b[i].heatFlux = 0.0f;
        b[i].htc = 0.0f;
    }
    return b;
}

inline DomainBoundaryConditions fromFlatArrays(
    const uint32_t types[6],
    const float temperaturesC[6],
    const float heatFluxes[6],
    const float heatTransferCoefficients[6],
    const float inflowTemperaturesC[6]) {
    DomainBoundaryConditions b{};
    for (size_t i = 0; i < 6; ++i) {
        b[i].type = static_cast<BoundaryType>(types[i]);
        b[i].temperatureC = temperaturesC[i];
        b[i].heatFlux = heatFluxes[i];
        b[i].htc = heatTransferCoefficients[i];
        b[i].inflowTemperatureC = inflowTemperaturesC[i];
    }
    return b;
}

} // namespace heat
