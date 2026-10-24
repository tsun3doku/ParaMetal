#pragma once

#include "VEMMeshBuilder.hpp"

#include <cstdint>
#include <vector>

namespace vem {

struct VEMCellInputs {
    std::vector<float> conductivity;
    std::vector<float> capacityOverDt;
    std::vector<float> sourceAdditions;
    std::vector<float> fixedTemperatures;
};

struct VEMCellRemap {
    std::vector<uint32_t> fragmentToCell;
    std::vector<uint32_t> cellFragmentOffsets;
    std::vector<uint32_t> cellFragments;
};

struct VEMAgglomerationResult {
    VEMMesh mesh;
    VEMCellInputs inputs;
    VEMCellRemap remap;
};

class VEMAgglomerator {
public:
    static bool build(
        VEMMesh mesh,
        const VEMCellInputs& fragmentInputs,
        VEMAgglomerationResult& result);

    static bool remapInputs(
        const VEMCellRemap& remap,
        const VEMCellInputs& fragmentInputs,
        VEMCellInputs& cellInputs);
};

} // namespace vem
