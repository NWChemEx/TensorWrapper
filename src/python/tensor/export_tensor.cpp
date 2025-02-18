#include "export_tensor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper {

void export_tensor(py_module_reference m) {
    py_class_type<Tensor>(m, "Tensor")
      .def(pybind11::init<>())
      .def(pybind11::init([](numpy_t& array) {
          auto rank = array.ndim();
          parallelzone::runtime::RuntimeView rv;
          auto palloc = alloc_t::make_eigen_allocator(rank, rv);

          auto pbuffer = palloc->(layout);
      }))
      .def("rank", &Tensor::rank)
      .def(pybind11::self == pybind11::self)
      .def(pybind11::self != pybind11::self)
      .def("__str__", [](Tensor& self) { return self.to_string(); });

    m.def("to_ndarray", [](Tensor& t) {
        t.buffer().data();
    });
}

} // namespace tensorwrapper