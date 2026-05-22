#include "nodegraph/NodeGraphCoreTypes.hpp"

#include <Python.h>
#ifdef slots
#undef slots
#endif

#include <pybind11/pybind11.h>

namespace py = pybind11;

void bindIds(py::module_& m) {
    py::class_<NodeGraphNodeId>(m, "NodeId")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def_readwrite("value", &NodeGraphNodeId::value)
        .def("is_valid", &NodeGraphNodeId::isValid)
        .def("__eq__", [](const NodeGraphNodeId& a, const NodeGraphNodeId& b) { return a == b; })
        .def("__hash__", [](const NodeGraphNodeId& id) { return std::hash<uint32_t>{}(id.value); })
        .def("__int__", [](const NodeGraphNodeId& id) { return id.value; })
        .def("__repr__", [](const NodeGraphNodeId& id) {
            return "NodeId(" + std::to_string(id.value) + ")";
        });
    py::implicitly_convertible<uint32_t, NodeGraphNodeId>();

    py::class_<NodeGraphSocketId>(m, "SocketId")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def_readwrite("value", &NodeGraphSocketId::value)
        .def("is_valid", &NodeGraphSocketId::isValid)
        .def("__eq__", [](const NodeGraphSocketId& a, const NodeGraphSocketId& b) { return a == b; })
        .def("__hash__", [](const NodeGraphSocketId& id) { return std::hash<uint32_t>{}(id.value); })
        .def("__int__", [](const NodeGraphSocketId& id) { return id.value; })
        .def("__repr__", [](const NodeGraphSocketId& id) {
            return "SocketId(" + std::to_string(id.value) + ")";
        });
    py::implicitly_convertible<uint32_t, NodeGraphSocketId>();

    py::class_<NodeGraphEdgeId>(m, "EdgeId")
        .def(py::init<>())
        .def(py::init<uint32_t>())
        .def_readwrite("value", &NodeGraphEdgeId::value)
        .def("is_valid", &NodeGraphEdgeId::isValid)
        .def("__eq__", [](const NodeGraphEdgeId& a, const NodeGraphEdgeId& b) { return a == b; })
        .def("__hash__", [](const NodeGraphEdgeId& id) { return std::hash<uint32_t>{}(id.value); })
        .def("__int__", [](const NodeGraphEdgeId& id) { return id.value; })
        .def("__repr__", [](const NodeGraphEdgeId& id) {
            return "EdgeId(" + std::to_string(id.value) + ")";
        });
    py::implicitly_convertible<uint32_t, NodeGraphEdgeId>();
}
