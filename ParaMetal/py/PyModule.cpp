#include "PyBridge.hpp"

#include "nodegraph/NodeGraph.hpp"
#include "nodegraph/NodeGraphEditor.hpp"

#include <pybind11/embed.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

// Forward-declare binding functions
void bindIds(py::module_& m);
void bindParams(py::module_& m);
void bindRegistry(py::module_& m);
void bindGraph(py::module_& m);

PYBIND11_EMBEDDED_MODULE(parametal, m) {
    m.doc() = "ParaMetal Python API";

    bindIds(m);
    bindParams(m);
    bindRegistry(m);
    bindGraph(m);

    m.def("get_graph", []() {
        NodeGraph* graph = pybridge::getGraph();
        if (!graph) {
            throw std::runtime_error("No graph available");
        }
        return graph;
    }, py::return_value_policy::reference);

    m.def("default_graph", []() {
        NodeGraph* graph = pybridge::getGraph();
        if (!graph) {
            throw std::runtime_error("No graph available");
        }

        NodeGraphEditor editor(*graph);
        editor.resetToDefaultGraph();
    });
}
