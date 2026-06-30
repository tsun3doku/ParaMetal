#include "PyGraphTypes.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bindGraph(py::module_& m) {
    py::class_<PySocket>(m, "Socket")
        .def(py::init<>())
        .def_property_readonly("name", &PySocket::name)
        .def_property_readonly("type", &PySocket::valueType)
        .def_property_readonly("node_id", [](const PySocket& s) { return s.nodeId; })
        .def_property_readonly("socket_id", [](const PySocket& s) { return s.socketId; })
        .def("__repr__", [](const PySocket& s) {
            return "Socket(name='" + s.name() + "')";
        });

    py::class_<PyEdge>(m, "Edge")
        .def(py::init<>())
        .def_property_readonly("id",          &PyEdge::edge_id)
        .def_property_readonly("from_node",   &PyEdge::from_node)
        .def_property_readonly("from_socket", &PyEdge::from_socket)
        .def_property_readonly("to_node",     &PyEdge::to_node)
        .def_property_readonly("to_socket",   &PyEdge::to_socket)
        .def("__repr__", [](const PyEdge& e) {
            PyNode fn = e.from_node();
            PyNode tn = e.to_node();
            PySocket fs = e.from_socket();
            PySocket ts = e.to_socket();
            return "Edge(" + fn.name() + "." + fs.name() +
                   " -> " + tn.name() + "." + ts.name() + ")";
        });

    py::class_<PyNode>(m, "Node")
        .def(py::init<>())
        .def_property_readonly("name", &PyNode::name)
        .def_property_readonly("type", &PyNode::type_id)
        .def("get", &PyNode::get, py::arg("param_name"))
        .def("set", &PyNode::set, py::arg("param_name"), py::arg("value"))
        .def("input", &PyNode::input, py::arg("name"))
        .def("output", &PyNode::output, py::arg("name"))
        .def_property_readonly("inputs", &PyNode::inputs)
        .def_property_readonly("outputs", &PyNode::outputs)
        .def("__repr__", [](const PyNode& n) {
            return "Node(name='" + n.name() + "', type='" + n.type_id() + "')";
        });

    py::class_<NodeGraph>(m, "Graph")
        .def("add_node", [](NodeGraph& self, const std::string& typeId, const std::string& name, float x, float y) {
            NodeGraphNodeId id = self.addNode(typeId, name, x, y);
            if (!id.isValid()) {
                throw py::value_error("Unknown node type: '" + typeId + "'. Use registry() to list valid types.");
            }
            return PyNode(&self, id);
        }, py::arg("type"), py::arg("name") = "", py::arg("x") = 0.0f, py::arg("y") = 0.0f)
        .def("remove_node", &NodeGraph::removeNode)
        .def("get_node", [](NodeGraph& self, const std::string& name) {
            NodeGraphState state = self.state();
            for (const auto& [id, node] : state.nodes) {
                if (node.title == name) {
                    return PyNode(&self, node.id);
                }
            }
            throw py::key_error("Node not found: " + name);
        }, py::arg("name"))
        .def_property_readonly("nodes", [](NodeGraph& self) {
            std::vector<PyNode> result;
            NodeGraphState state = self.state();
            for (const auto& [id, node] : state.nodes) {
                result.emplace_back(&self, node.id);
            }
            return result;
        })
        .def_property_readonly("edges", [](NodeGraph& self) {
            std::vector<PyEdge> result;
            NodeGraphState state = self.state();
            for (const auto& [id, edge] : state.edges) {
                result.emplace_back(&self, edge.id);
            }
            return result;
        })
        .def("connect", [](NodeGraph& self, const PySocket& outSocket, const PySocket& inSocket) {
            std::string error;
            bool ok = self.connectSockets(outSocket.nodeId, outSocket.socketId, inSocket.nodeId, inSocket.socketId, error);
            if (!ok) {
                throw std::runtime_error("Connection failed: " + error);
            }
        }, py::arg("output"), py::arg("input"))
        .def("disconnect", [](NodeGraph& self, const PyEdge& edge) {
            if (!self.removeConnection(edge.edge_id())) {
                throw std::runtime_error("disconnect failed: edge not found");
            }
        }, py::arg("edge"))
        .def_property_readonly("registry", [](NodeGraph& self) -> NodeGraphRegistry& {
            return self.getRegistry();
        }, py::return_value_policy::reference)
        .def("__repr__", [](NodeGraph& self) {
            NodeGraphState state = self.state();
            return "Graph(nodes=" + std::to_string(state.nodes.size()) +
                   ", edges=" + std::to_string(state.edges.size()) + ")";
        });
}
