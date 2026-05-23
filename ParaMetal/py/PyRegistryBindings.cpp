#include "nodegraph/NodeGraphRegistry.hpp"
#include "nodegraph/NodeGraphTypes.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindRegistry(py::module_& m) {
    py::enum_<NodeGraphValueType>(m, "ValueType")
        .value("None", NodeGraphValueType::None)
        .value("Mesh", NodeGraphValueType::Mesh)
        .value("Volume", NodeGraphValueType::Volume)
        .value("Field", NodeGraphValueType::Field)
        .value("Vector3", NodeGraphValueType::Vector3)
        .value("ScalarFloat", NodeGraphValueType::ScalarFloat)
        .value("ScalarInt", NodeGraphValueType::ScalarInt)
        .value("ScalarBool", NodeGraphValueType::ScalarBool);

    py::enum_<NodeGraphSocketDirection>(m, "SocketDirection")
        .value("Input", NodeGraphSocketDirection::Input)
        .value("Output", NodeGraphSocketDirection::Output);

    py::enum_<NodeGraphNodeCategory>(m, "NodeCategory")
        .value("Model", NodeGraphNodeCategory::Model)
        .value("PointSurface", NodeGraphNodeCategory::PointSurface)
        .value("Meshing", NodeGraphNodeCategory::Meshing)
        .value("System", NodeGraphNodeCategory::System)
        .value("Custom", NodeGraphNodeCategory::Custom);

    py::class_<NodeGraphSocketContract>(m, "SocketContract")
        .def_readwrite("produced_payload_type", &NodeGraphSocketContract::producedPayloadType);

    py::class_<NodeSocketSignature>(m, "SocketSignature")
        .def_readwrite("name", &NodeSocketSignature::name)
        .def_readwrite("direction", &NodeSocketSignature::direction)
        .def_readwrite("value_type", &NodeSocketSignature::valueType)
        .def_readwrite("contract", &NodeSocketSignature::contract)
        .def_readwrite("variadic", &NodeSocketSignature::variadic);

    py::class_<NodeTypeDefinition>(m, "NodeTypeDefinition")
        .def_readwrite("id", &NodeTypeDefinition::id)
        .def_readwrite("display_name", &NodeTypeDefinition::displayName)
        .def_readwrite("category", &NodeTypeDefinition::category)
        .def_readwrite("sockets", &NodeTypeDefinition::sockets)
        .def_readwrite("parameters", &NodeTypeDefinition::parameters);

    py::class_<NodeGraphRegistry>(m, "Registry")
        .def("find_node_type", &NodeGraphRegistry::findNodeType, py::return_value_policy::reference)
        .def_property_readonly("all_node_types", &NodeGraphRegistry::allNodeTypes, py::return_value_policy::reference);
}
