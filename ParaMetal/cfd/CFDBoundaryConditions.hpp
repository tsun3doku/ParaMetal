#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

#include "CFDGpuStructs.hpp"

namespace cfd {

enum class BoundaryType : uint8_t {
    NoSlipWall,
    VelocityInlet,
    PressureOutlet
};

struct BoundaryCondition {
    BoundaryType type = BoundaryType::NoSlipWall;
    glm::vec3 velocity{0.0f};
};

using DomainBoundaryConditions =
    std::array<BoundaryCondition, 6>;

constexpr uint32_t DomainBoundaryCount = 6;
constexpr uint32_t EmbeddedWallBoundary = 6;

inline glm::vec3 getDomainBoxNormal(uint32_t domainFace) {
    switch (domainFace) {
        case 0: return {-1.0f,  0.0f,  0.0f};
        case 1: return { 1.0f,  0.0f,  0.0f};
        case 2: return { 0.0f, -1.0f,  0.0f};
        case 3: return { 0.0f,  1.0f,  0.0f};
        case 4: return { 0.0f,  0.0f, -1.0f};
        case 5: return { 0.0f,  0.0f,  1.0f};
        default: return {0.0f,  0.0f,  0.0f};
    }
}

inline BoundaryConditionGpu toGpu(const BoundaryCondition& bc) {
    BoundaryConditionGpu g{};
    g.type = static_cast<uint32_t>(bc.type);
    g.velocity = bc.velocity;
    return g;
}

inline DomainBoundaryConditions createDefaultDomainBoundaries(const glm::vec3& initialVelocity = glm::vec3(0.0f)) {
    DomainBoundaryConditions b{};
    b[0].type = BoundaryType::VelocityInlet;
    b[0].velocity = initialVelocity;
    b[1].type = BoundaryType::PressureOutlet;
    b[2].type = BoundaryType::NoSlipWall;
    b[3].type = BoundaryType::NoSlipWall;
    b[4].type = BoundaryType::NoSlipWall;
    b[5].type = BoundaryType::NoSlipWall;
    return b;
}

inline DomainBoundaryConditions fromFlatArrays(
    const uint32_t domainBoundaryTypes[DomainBoundaryCount],
    const glm::vec3 domainBoundaryVelocities[DomainBoundaryCount]) {
    DomainBoundaryConditions b{};
    for (uint32_t i = 0; i < DomainBoundaryCount; ++i) {
        b[i].type = static_cast<BoundaryType>(domainBoundaryTypes[i]);
        b[i].velocity = domainBoundaryVelocities[i];
    }
    return b;
}

inline std::array<BoundaryCondition, 7> toFullBoundaryArray(const DomainBoundaryConditions& domainBcs) {
    std::array<BoundaryCondition, 7> full{};
    for (size_t i = 0; i < 6; ++i) {
        full[i] = domainBcs[i];
    }
    full[EmbeddedWallBoundary].type = BoundaryType::NoSlipWall;
    full[EmbeddedWallBoundary].velocity = glm::vec3(0.0f);
    return full;
}

inline std::array<BoundaryConditionGpu, 7> toGpuArray(const DomainBoundaryConditions& domainBcs) {
    std::array<BoundaryConditionGpu, 7> full{};
    for (size_t i = 0; i < 6; ++i) {
        full[i] = toGpu(domainBcs[i]);
    }
    full[EmbeddedWallBoundary] = toGpu({BoundaryType::NoSlipWall, glm::vec3(0.0f)});
    return full;
}

} // namespace cfd
