#include "arcface_module.hpp"
#include "ndarray_converter.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/stl_bind.h>
#include <pybind11/stl.h>
namespace py = pybind11;



PYBIND11_MODULE(arcface_module, m) {
    NDArrayConverter::init_numpy();
    py::bind_vector<std::vector<int>>(m, "VectorFloat", py::buffer_protocol());
    py::bind_vector<std::vector<std::vector<float>>>(m, "vVectorFloat");
    py::class_<ArcFace>(m, "ArcFace")
        .def(py::init<const std::string &>())
        .def("extract", &ArcFace::extract);
}

