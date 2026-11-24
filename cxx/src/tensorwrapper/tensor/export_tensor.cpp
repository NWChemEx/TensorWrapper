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
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/concepts/floating_point.hpp>
#include <tensorwrapper/tensor/tensor.hpp>

namespace tensorwrapper {

namespace {

template<typename T>
auto get_desc_() -> decltype(py::format_descriptor<float>::format()) {
    if constexpr(std::is_same_v<T, float>)
        return py::format_descriptor<float>::format();
    else if constexpr(std::is_same_v<T, double>)
        return py::format_descriptor<double>::format();
    else if constexpr(std::is_same_v<T, long double>)
        return py::format_descriptor<long double>::format();
    else
        throw std::runtime_error("Unsupported floating point type!");
}

struct GetBufferDataKernel {
    using size_type  = std::size_t;
    using shape_type = shape::Smooth;

    GetBufferDataKernel(size_type rank, shape_type& smooth_shape) :
      m_rank(rank), m_psmooth_shape(&smooth_shape) {}

    template<concepts::FloatingPoint FloatType>
    py::buffer_info operator()(std::span<FloatType> buffer) {
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
        return py::buffer_info(ptr, nbytes, desc, rank, shape, strides);
    }

    size_type m_rank;
    shape_type* m_psmooth_shape;
};

template<typename FloatType>
Tensor make_tensor_(py::buffer_info& info) {
    if(info.format != py::format_descriptor<FloatType>::format())
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

/** @brief Makes a Tensor whose elements are the objects in @p arr.
 *
 *  @tparam FloatType The type each element of @p arr is cast to.
 *
 *  @param[in] arr A NumPy array of Python objects.
 *
 *  @return A Tensor with the shape and elements of @p arr.
 *
 *  @throw py::type_error if an element of @p arr is not a @p FloatType.
 *                        Strong throw guarantee.
 */
template<typename FloatType>
Tensor make_tensor_from_objects_(const py::array& arr) {
    std::vector<std::size_t> dims(arr.shape(), arr.shape() + arr.ndim());
    shape::Smooth shape(dims.begin(), dims.end());

    std::vector<FloatType> data;
    data.reserve(arr.size());
    try {
        for(auto x : arr.attr("flat")) data.push_back(x.cast<FloatType>());
    } catch(const py::cast_error&) {
        throw py::type_error("The elements of a Tensor must all be the same "
                             "type!");
    }
    auto pBuffer = std::make_unique<buffer::Contiguous>(std::move(data), shape);

    return Tensor(shape, std::move(pBuffer));
}

/** @brief Makes a Tensor from a (possibly nested) sequence of scalars.
 *
 *  The element type of the Tensor is set by the first scalar. At present the
 *  only scalars supported this way are the sigma types which sigma's Python
 *  module binds: intervals and triplexes.
 *
 *  @param[in] obj A scalar, a nested sequence of scalars, or a NumPy array of
 *                 Python objects.
 *
 *  @return A Tensor with the shape and elements of @p obj.
 *
 *  @throw std::runtime_error if the scalars are not of a supported type.
 *                            Strong throw guarantee.
 *  @throw py::type_error if the scalars are not all of the same type. Strong
 *                        throw guarantee.
 */
Tensor make_tensor_from_objects(py::object obj) {
#ifdef ENABLE_SIGMA
    auto np       = py::module_::import("numpy");
    py::array arr = np.attr("asarray")(obj, py::arg("dtype") = "object");
    if(arr.size() > 0) {
        py::object first = arr.attr("flat")[py::int_(0)];
        if(py::isinstance<types::idouble>(first))
            return make_tensor_from_objects_<types::idouble>(arr);
        if(py::isinstance<types::ifloat>(first))
            return make_tensor_from_objects_<types::ifloat>(arr);
        if(py::isinstance<types::tpdouble>(first))
            return make_tensor_from_objects_<types::tpdouble>(arr);
        if(py::isinstance<types::tpfloat>(first))
            return make_tensor_from_objects_<types::tpfloat>(arr);
    }
#endif
    throw std::runtime_error(
      "Incompatible format: expected a double array or sigma scalars!");
}

/// Copies the elements of a buffer into a new NumPy array
struct ToNumpyKernel {
    template<concepts::FloatingPoint FloatType>
    py::array operator()(std::span<FloatType> buffer) {
        using clean_type = std::decay_t<FloatType>;

        if constexpr(std::is_floating_point_v<clean_type>) {
            return py::array_t<clean_type>(m_shape, buffer.data());
        } else if constexpr(types::is_interval_v<clean_type> ||
                            types::is_triplex_v<clean_type>) {
            // Registers the sigma types with pybind11, so they can be cast
            py::module_::import("sigma");
            auto np = py::module_::import("numpy");
            py::tuple shape(m_shape.size());
            for(std::size_t i = 0; i < m_shape.size(); ++i)
                shape[i] = m_shape[i];
            py::array rv = np.attr("empty")(shape, py::arg("dtype") = "object");
            auto flat    = rv.attr("flat");
            for(std::size_t i = 0; i < buffer.size(); ++i)
                flat[py::int_(i)] = py::cast(buffer[i]);
            return rv;
        } else {
            throw std::runtime_error(
              "The tensor's element type has no Python binding!");
        }
    }

    std::vector<std::size_t> m_shape;
};

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

Tensor make_tensor(py::buffer b) {
    py::buffer_info info = b.request();
    if(info.format == py::format_descriptor<double>::format())
        return make_tensor_<double>(info);
    else if(info.format == "O") // A NumPy array of Python objects
        return make_tensor_from_objects(b);
    else
        throw std::runtime_error(
          "Incompatible format: expected a double array!");
}

void export_tensor(py_module_reference m) {
    py_class_type<Tensor>(m, "Tensor", py::buffer_protocol())
      .def(py::init<>())
      .def(py::init([](py::buffer b) { return make_tensor(b); }))
      .def(py::init([](py::object o) { return make_tensor_from_objects(o); }))
      .def("rank", &Tensor::rank)
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__str__", [](Tensor& self) { return self.to_string(); })
      // NumPy falls back to this when the buffer protocol fails, i.e., for
      // element types NumPy can not describe, like sigma's intervals
      .def(
        "__array__",
        [](Tensor& self, py::object dtype, py::object) {
            auto& buffer            = buffer::make_contiguous(self.buffer());
            const auto smooth_shape = buffer.layout().shape().as_smooth();
            ToNumpyKernel kernel;
            for(std::size_t i = 0; i < smooth_shape.rank(); ++i)
                kernel.m_shape.push_back(smooth_shape.extent(i));
            py::object rv = buffer::visit_contiguous_buffer(kernel, buffer);
            return dtype.is_none() ? rv : rv.attr("astype")(dtype);
        },
        py::arg("dtype") = py::none(), py::arg("copy") = py::none())
      .def_buffer([](Tensor& t) {
          auto pbuffer = dynamic_cast<buffer::Contiguous*>(&t.buffer());
          if(pbuffer == nullptr)
              throw std::runtime_error("Expected buffer to be contiguous");
          return make_buffer_info(*pbuffer);
      });
}

} // namespace tensorwrapper
