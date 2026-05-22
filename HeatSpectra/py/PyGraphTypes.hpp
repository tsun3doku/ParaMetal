#pragma once

// Python binding-only header. Do NOT include from Qt UI files.
#include <Python.h>
#ifdef slots
#undef slots
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nodegraph/NodeGraphBridge.hpp"
#include "nodegraph/NodeGraphTypes.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"

#include <string>
#include <vector>

namespace py = pybind11;

py::object paramValueToPy(const NodeGraphParamValue& value);
NodeGraphParamValue pyToParamValue(uint32_t paramId, NodeGraphParamType type, py::object value);

struct PySocket {
    NodeGraphBridge* bridge = nullptr;
    NodeGraphNodeId nodeId{};
    NodeGraphSocketId socketId{};

    PySocket() = default;
    PySocket(NodeGraphBridge* b, NodeGraphNodeId n, NodeGraphSocketId s);

    std::string name() const;
    NodeGraphValueType valueType() const;
};

struct PyEdge {
    NodeGraphBridge* bridge = nullptr;
    NodeGraphEdgeId edgeId{};

    PyEdge() = default;
    PyEdge(NodeGraphBridge* b, NodeGraphEdgeId id);

    NodeGraphNodeId from_node() const;
    NodeGraphSocketId from_socket() const;
    NodeGraphNodeId to_node() const;
    NodeGraphSocketId to_socket() const;

private:
    NodeGraphEdge getEdge() const;
};

struct PyNode {
    NodeGraphBridge* bridge = nullptr;
    NodeGraphNodeId nodeId{};

    PyNode() = default;
    PyNode(NodeGraphBridge* b, NodeGraphNodeId id);

    std::string name() const;
    std::string type_id() const;
    py::object get(const std::string& paramName) const;
    void set(const std::string& paramName, py::object value);
    PySocket input(const std::string& name) const;
    PySocket output(const std::string& name) const;
    std::vector<PySocket> inputs() const;
    std::vector<PySocket> outputs() const;
};
