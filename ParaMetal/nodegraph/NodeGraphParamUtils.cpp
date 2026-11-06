#include "NodeGraphParamUtils.hpp"

#include <utility>

NodeGraphParamDefinition makeEnumParamDefinition(
    uint32_t id,
    const char* name,
    const char* defaultValue,
    std::vector<std::string> enumOptions) {
    NodeGraphParamDefinition definition{};
    definition.id = id;
    definition.name = name;
    definition.type = NodeGraphParamType::Enum;
    definition.defaultStringValue = defaultValue;
    definition.enumOptions = std::move(enumOptions);
    return definition;
}

NodeGraphParamValue makeEnumParamValue(std::string value) {
    NodeGraphParamValue parameter{};
    parameter.type = NodeGraphParamType::Enum;
    parameter.enumValue = std::move(value);
    return parameter;
}

NodeGraphParamDefinition makeStructParamDefinition(
    uint32_t id, const char* name, std::vector<NodeGraphParamField> fields) {
    NodeGraphParamDefinition definition{};
    definition.id = id;
    definition.name = name;
    definition.type = NodeGraphParamType::Struct;
    definition.fields = std::move(fields);
    return definition;
}

NodeGraphParamDefinition makeArrayParamDefinition(
    uint32_t id, const char* name, NodeGraphParamDefinition elementDefinition) {
    NodeGraphParamDefinition definition{};
    definition.id = id;
    definition.name = name;
    definition.type = NodeGraphParamType::Array;
    definition.elementDefinition = std::make_shared<NodeGraphParamDefinition>(std::move(elementDefinition));
    return definition;
}

bool tryGetParamInt(const NodeGraphParamValue& value, int64_t& outValue) {
    if (value.type != NodeGraphParamType::Int) {
        return false;
    }
    outValue = value.intValue;
    return true;
}

bool tryGetParamString(const NodeGraphParamValue& value, std::string& outValue) {
    if (value.type != NodeGraphParamType::String) {
        return false;
    }
    outValue = value.stringValue;
    return true;
}

bool tryGetParamEnum(const NodeGraphParamValue& value, std::string& outValue) {
    if (value.type != NodeGraphParamType::Enum) {
        return false;
    }
    outValue = value.enumValue;
    return true;
}
