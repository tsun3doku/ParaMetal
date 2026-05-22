#include "nodegraph/NodeGraphTypes.hpp"
#include "nodegraph/NodeGraphParamUtils.hpp"

#include <Python.h>
#ifdef slots
#undef slots
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindParams(py::module_& m) {
    py::enum_<NodeGraphParamType>(m, "ParamType")
        .value("Float", NodeGraphParamType::Float)
        .value("Int", NodeGraphParamType::Int)
        .value("Bool", NodeGraphParamType::Bool)
        .value("String", NodeGraphParamType::String)
        .value("Enum", NodeGraphParamType::Enum)
        .value("Struct", NodeGraphParamType::Struct)
        .value("Array", NodeGraphParamType::Array);

    py::class_<NodeGraphParamValue>(m, "ParamValue")
        .def(py::init<>())
        .def_readwrite("id", &NodeGraphParamValue::id)
        .def_readwrite("type", &NodeGraphParamValue::type)
        .def_readwrite("float_value", &NodeGraphParamValue::floatValue)
        .def_readwrite("int_value", &NodeGraphParamValue::intValue)
        .def_readwrite("bool_value", &NodeGraphParamValue::boolValue)
        .def_readwrite("string_value", &NodeGraphParamValue::stringValue)
        .def_readwrite("enum_value", &NodeGraphParamValue::enumValue)
        .def("__repr__", [](const NodeGraphParamValue& v) {
            switch (v.type) {
                case NodeGraphParamType::Float: return "ParamValue(float=" + std::to_string(v.floatValue) + ")";
                case NodeGraphParamType::Int: return "ParamValue(int=" + std::to_string(v.intValue) + ")";
                case NodeGraphParamType::Bool: return "ParamValue(bool=" + std::string(v.boolValue ? "True" : "False") + ")";
                case NodeGraphParamType::String: return "ParamValue(string='" + v.stringValue + "')";
                case NodeGraphParamType::Enum: return "ParamValue(enum='" + v.enumValue + "')";
                default: return std::string("ParamValue()");
            }
        });

    py::class_<NodeGraphParamDefinition>(m, "ParamDefinition")
        .def_readwrite("id", &NodeGraphParamDefinition::id)
        .def_readwrite("name", &NodeGraphParamDefinition::name)
        .def_readwrite("type", &NodeGraphParamDefinition::type)
        .def_readwrite("default_float", &NodeGraphParamDefinition::defaultFloatValue)
        .def_readwrite("default_int", &NodeGraphParamDefinition::defaultIntValue)
        .def_readwrite("default_bool", &NodeGraphParamDefinition::defaultBoolValue)
        .def_readwrite("default_string", &NodeGraphParamDefinition::defaultStringValue)
        .def_readwrite("is_action", &NodeGraphParamDefinition::isAction)
        .def_readwrite("enum_options", &NodeGraphParamDefinition::enumOptions);

    py::class_<NodeGraphParamField>(m, "ParamField")
        .def_readwrite("name", &NodeGraphParamField::name);

    py::class_<NodeGraphParamFieldValue>(m, "ParamFieldValue")
        .def_readwrite("name", &NodeGraphParamFieldValue::name);
}
