/*
 * Copyright 2025 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "export_tensor.hpp"
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper {

template<typename FloatType>
auto make_buffer_info(buffer::Contiguous<FloatType>& buffer) {
    using size_type       = std::size_t;
    constexpr auto nbytes = sizeof(FloatType);
    const auto desc       = pybind11::format_descriptor<FloatType>::format();
    const auto rank       = buffer.rank();

    const auto smooth_shape = buffer.layout().shape().as_smooth();

    std::vector<size_type> shape(rank);
    std::vector<size_type> strides(rank);
    for(size_type rank_i = 0; rank_i < rank; ++rank_i) {
        shape[rank_i]      = smooth_shape.extent(rank_i);
        size_type stride_i = 1;
        for(size_type mode_i = rank_i + 1; mode_i < rank; ++mode_i)
            stride_i *= smooth_shape.extent(mode_i);
        strides[rank_i] = stride_i * nbytes;
    }
    return pybind11::buffer_info(buffer.get_mutable_data(), nbytes, desc, rank,
                                 shape, strides);
}

auto make_tensor(pybind11::buffer b) {
    pybind11::buffer_info info = b.request();
    if(info.format != pybind11::format_descriptor<double>::format())
        throw std::runtime_error(
          "Incompatible format: expected a double array!");

    std::vector<std::size_t> dims(info.ndim);
    for(auto i = 0; i < info.ndim; ++i) { dims[i] = info.shape[i]; }

    parallelzone::runtime::RuntimeView rv = {};
    allocator::Eigen<double> allocator(rv);
    shape::Smooth matrix_shape{dims.begin(), dims.end()};
    layout::Physical matrix_layout(matrix_shape);
    auto pBuffer = allocator.allocate(matrix_layout);

    auto n_elements = std::accumulate(dims.begin(), dims.end(), 1,
                                      std::multiplies<std::size_t>());
    auto pData      = static_cast<double*>(info.ptr);
    for(auto i = 0; i < n_elements; ++i) pBuffer->set_data(i, pData[i]);

    return Tensor(matrix_shape, std::move(pBuffer));
}

void export_tensor(py_module_reference m) {
    py_class_type<Tensor>(m, "Tensor", pybind11::buffer_protocol())
      .def(pybind11::init<>())
      .def(pybind11::init([](pybind11::buffer b) { return make_tensor(b); }))
      .def("rank", &Tensor::rank)
      .def(pybind11::self == pybind11::self)
      .def(pybind11::self != pybind11::self)
      .def("__str__", [](Tensor& self) { return self.to_string(); })
      .def_buffer([](Tensor& t) {
          auto pbuffer = dynamic_cast<buffer::Contiguous<double>*>(&t.buffer());
          if(pbuffer == nullptr)
              throw std::runtime_error("Expected buffer to hold doubles");
          return make_buffer_info(*pbuffer);
      });
}

} // namespace tensorwrapper