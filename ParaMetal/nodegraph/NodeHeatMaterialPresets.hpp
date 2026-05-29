#pragma once

#include "NodeGraphTypes.hpp"
#include "domain/HeatData.hpp"
#include <string>
#include <vector>

#include "heat/HeatSystemPresets.hpp"

bool tryResolveHeatPresetId(const std::string& value, HeatMaterialPresetId& outId);
bool tryMakeMaterialBinding(
    const std::string& receiverText,
    const std::string& presetText,
    HeatMaterialBinding& outRow);
std::vector<HeatMaterialBinding> readMaterialBindings(const NodeGraphParamValue& value);
bool writeMaterialBindings(NodeGraphParamValue& value, const std::vector<HeatMaterialBinding>& rows);
