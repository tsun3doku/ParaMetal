#pragma once

#include "NodeGraphTypes.hpp"

#include <cstdint>
#include <string>
#include <vector>

NodeGraphParamDefinition makeEnumParamDefinition(uint32_t id, const char* name, const char* defaultValue, std::vector<std::string> enumOptions);
NodeGraphParamDefinition makeStructParamDefinition(uint32_t id, const char* name, std::vector<NodeGraphParamField> fields);
NodeGraphParamDefinition makeArrayParamDefinition(uint32_t id, const char* name, NodeGraphParamDefinition elementDefinition);
NodeGraphParamValue makeEnumParamValue(std::string value);

bool tryGetParamInt(const NodeGraphParamValue& value, int64_t& outValue);
bool tryGetParamString(const NodeGraphParamValue& value, std::string& outValue);
bool tryGetParamEnum(const NodeGraphParamValue& value, std::string& outValue);
