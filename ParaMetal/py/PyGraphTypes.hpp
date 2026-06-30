#pragma once

// Python binding-only header. Do NOT include from Qt UI files.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "nodegraph/NodeGraph.hpp"
#include "nodegraph/NodeGraphTypes.hpp"
#include "nodegraph/NodeGraphRegistry.hpp"

#include <string>
#include <vector>

namespace py = pybind11;

py::object paramValueToPy(const NodeGraphParamValue& value);
NodeGraphParamValue pyToParamValue(uint32_t paramId, NodeGraphParamType type, py::object value);

struct PySocket {
    NodeGraph* bridge = nullptr;
    NodeGraphNodeId nodeId{};
    NodeGraphSocketId socketId{};

    PySocket() = default;
    PySocket(NodeGraph* b, NodeGraphNodeId n, NodeGraphSocketId s);

    std::string name() const;
    NodeGraphValueType valueType() const;
};

struct PyNode {
    NodeGraph* bridge = nullptr;
    NodeGraphNodeId nodeId{};

    PyNode() = default;
    PyNode(NodeGraph* b, NodeGraphNodeId id);

    std::string name() const;
    std::string type_id() const;
    py::object get(const std::string& paramName) const;
    void set(const std::string& paramName, py::object value);
    PySocket input(const std::string& name) const;
    PySocket output(const std::string& name) const;
    std::vector<PySocket> inputs() const;
    std::vector<PySocket> outputs() const;
};

struct PyEdge {
    NodeGraph* bridge = nullptr;
    NodeGraphEdgeId edgeId{};

    PyEdge() = default;
    PyEdge(NodeGraph* b, NodeGraphEdgeId id);

    NodeGraphEdgeId edge_id() const { return edgeId; }
    PyNode from_node() const;
    PySocket from_socket() const;
    PyNode to_node() const;
    PySocket to_socket() const;

private:
    NodeGraphEdge getEdge() const;
};
