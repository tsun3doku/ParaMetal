#pragma once

#include <cstdint>

//                                                          [ HashValues / HashDomain
//                                                            -Shared domain specific hash identity container
//                                                            -Each domain represents a special case:
//                                                              Full       - whole object identity
//                                                              Geometry   - mesh/points/domain shape
//                                                              Thermal    - material, temp, boundary condition
//                                                              Simulation - everything to validate sim history
//                                                              Display    - visual only settings
//                                                            -Payloads, packages and products carry hash values  ]

enum class HashDomain : uint8_t {
    Full,
    Geometry,
    Thermal,
    Simulation,
    Display
};

struct HashValues {
    uint64_t full = 0;
    uint64_t geometry = 0;
    uint64_t thermal = 0;
    uint64_t simulation = 0;
    uint64_t display = 0;

    uint64_t get(HashDomain domain) const {
        switch (domain) {
        case HashDomain::Full:       return full;
        case HashDomain::Geometry:   return geometry;
        case HashDomain::Thermal:    return thermal;
        case HashDomain::Simulation: return simulation;
        case HashDomain::Display:    return display;
        }
        return 0;
    }
};
