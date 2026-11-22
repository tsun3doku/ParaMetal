#pragma once

#include "NodeGraphTypes.hpp"

#include <initializer_list>

inline NodeSocketSignature makeInputSocket(
    const char* name,
    NodeGraphValueType valueType,
    bool variadic = false,
    bool required = true) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Input;
    signature.valueType = valueType;
    signature.variadic = variadic;
    signature.required = required;
    return signature;
}

inline NodeSocketSignature makeOutputSocket(
    const char* name,
    NodeGraphValueType valueType,
    uint8_t producedPayloadType,
    bool requiresRuntimePackage = false) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Output;
    signature.valueType = valueType;
    signature.contract.producedPayloadType = producedPayloadType;
    signature.contract.requiresRuntimePackage = requiresRuntimePackage;
    return signature;
}

inline NodeSocketSignature makePolymorphicInputSocket(
    const char* name,
    std::initializer_list<NodeGraphValueType> acceptedValueTypes,
    bool variadic = false,
    bool required = true) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Input;
    signature.valueType = NodeGraphValueType::None;
    signature.acceptedValueTypes = acceptedValueTypes;
    signature.variadic = variadic;
    signature.required = required;
    return signature;
}

inline NodeSocketSignature makePolymorphicOutputSocket(
    const char* name,
    std::initializer_list<NodeGraphValueType> acceptedValueTypes,
    uint8_t producedPayloadType,
    bool requiresRuntimePackage = false) {
    NodeSocketSignature signature{};
    signature.name = name;
    signature.direction = NodeGraphSocketDirection::Output;
    signature.valueType = NodeGraphValueType::None;
    signature.acceptedValueTypes = acceptedValueTypes;
    signature.contract.producedPayloadType = producedPayloadType;
    signature.contract.requiresRuntimePackage = requiresRuntimePackage;
    return signature;
}
