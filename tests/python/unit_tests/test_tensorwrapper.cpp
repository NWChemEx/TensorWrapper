#include "test_tensorwrapper.hpp"
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper::testing {

PYBIND11_MODULE(py_test_tensorwrapper, m) {
    auto m_testing = m.def_submodule("testing");
    get_scalar(m_testing);
    get_vector(m_testing);
    get_matrix(m_testing);
}

void get_scalar(pybind11::module_& m) {
    m.def("get_scalar", []() { return Tensor(42.0); });
}
void get_vector(pybind11::module_& m) {
    m.def("get_vector", []() { return Tensor{0.0, 1.0, 2.0, 3.0, 4.0}; });
}
void get_matrix(pybind11::module_& m) {
    m.def("get_matrix", []() { return Tensor{{1.0, 2.0}, {3.0, 4.0}}; });
}

} // namespace tensorwrapper::testing