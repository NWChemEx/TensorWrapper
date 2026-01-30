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
namespace {

template<typename T>
auto get_desc_() -> decltype(pybind11::format_descriptor<float>::format()) {
    if constexpr(std::is_same_v<T, float>)
        return pybind11::format_descriptor<float>::format();
    else if constexpr(std::is_same_v<T, double>)
        return pybind11::format_descriptor<double>::format();
    else if constexpr(std::is_same_v<T, long double>)
        return pybind11::format_descriptor<long double>::format();
    else
        throw std::runtime_error("Unsupported floating point type!");
}

struct GetBufferDataKernel {
    using size_type  = std::size_t;
    using shape_type = shape::Smooth;

    GetBufferDataKernel(size_type rank, shape_type& smooth_shape) :
      m_rank(rank), m_psmooth_shape(&smooth_shape) {}

    template<concepts::FloatingPoint FloatType>
    pybind11::buffer_info operator()(std::span<FloatType> buffer) {
        using clean_type = std::decay_t<FloatType>;

        // We have only tested with doubles at the moment.
        if constexpr(!std::is_same_v<clean_type, double>)
            throw std::runtime_error("Expected doubles in the buffer!");

        constexpr auto nbytes = sizeof(clean_type);

        const auto desc = get_desc_<clean_type>();
        const auto rank = m_rank;

        std::vector<size_type> shape(rank);
        std::vector<size_type> strides(rank);
        for(size_type rank_i = 0; rank_i < rank; ++rank_i) {
            shape[rank_i]      = m_psmooth_shape->extent(rank_i);
            size_type stride_i = 1;
            for(size_type mode_i = rank_i + 1; mode_i < rank; ++mode_i)
                stride_i *= m_psmooth_shape->extent(mode_i);
            strides[rank_i] = stride_i * nbytes;
        }
        auto* ptr = const_cast<clean_type*>(buffer.data());
        return pybind11::buffer_info(ptr, nbytes, desc, rank, shape, strides);
    }

    size_type m_rank;
    shape_type* m_psmooth_shape;
};

template<typename FloatType>
Tensor make_tensor_(pybind11::buffer_info& info) {
    if(info.format != pybind11::format_descriptor<FloatType>::format())
        throw std::runtime_error(
          "Incompatible format: expected a float array!");

    // Work out physical layout of tensor
    std::vector<std::size_t> dims(info.ndim);
    for(auto i = 0; i < info.ndim; ++i) { dims[i] = info.shape[i]; }
    shape::Smooth shape(dims.begin(), dims.end());
    layout::Physical layout(shape);

    // Fill in Buffer object
    auto n_elements = shape.size();
    std::vector<FloatType> data(n_elements);
    auto pData = static_cast<FloatType*>(info.ptr);
    std::copy(pData, pData + n_elements, data.begin());
    auto pBuffer = std::make_unique<buffer::Contiguous>(data, shape);

    return Tensor(shape, std::move(pBuffer));
}

} // namespace

auto make_buffer_info(buffer::Contiguous& buffer) {
    const auto rank         = buffer.rank();
    const auto smooth_shape = buffer.layout().shape().as_smooth();
    std::vector<std::size_t> extents(rank);
    for(std::size_t i = 0; i < rank; ++i) extents[i] = smooth_shape.extent(i);
    shape::Smooth shape(extents.begin(), extents.end());
    GetBufferDataKernel kernel(rank, shape);
    return buffer::visit_contiguous_buffer(kernel, buffer);
}

Tensor make_tensor(pybind11::buffer b) {
    pybind11::buffer_info info = b.request();
    if(info.format == pybind11::format_descriptor<double>::format())
        return make_tensor_<double>(info);
    else
        throw std::runtime_error(
          "Incompatible format: expected a double array!");
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
          auto pbuffer = dynamic_cast<buffer::Contiguous*>(&t.buffer());
          if(pbuffer == nullptr)
              throw std::runtime_error("Expected buffer to be contiguous");
          return make_buffer_info(*pbuffer);
      });
}

} // namespace tensorwrapper
