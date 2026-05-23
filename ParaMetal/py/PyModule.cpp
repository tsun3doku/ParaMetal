#include "PyBridge.hpp"

#include "nodegraph/NodeGraphBridge.hpp"

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
        NodeGraphBridge* bridge = pybridge::getBridge();
        if (!bridge) {
            throw std::runtime_error("No graph bridge available");
        }
        return bridge;
    }, py::return_value_policy::reference);
}
