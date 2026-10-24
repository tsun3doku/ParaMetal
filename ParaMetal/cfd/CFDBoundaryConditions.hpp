#pragma once

#include <array>
#include <cstdint>
#include <glm/glm.hpp>

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

struct BoundaryConditionGpu {
    uint32_t type = 0;
    glm::vec3 velocity{0.0f};
};

constexpr uint32_t DomainBoundaryCount = 6;
constexpr uint32_t EmbeddedWallBoundary = 6;

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
