/*
 * Copyright 2022 NWChemEx-Project
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

#include "../buffer/detail_/ta_buffer_pimpl.hpp"
#include "../conversion/conversion.hpp"
#include <tensorwrapper/tensor/allocator/allocator.hpp>
#include <tensorwrapper/tensor/backends/tiledarray.hpp>
#include <tensorwrapper/tensor/detail_/pimpl.hpp>
#include <tensorwrapper/tensor/shapes/shape.hpp>

namespace tensorwrapper::tensor::backends {

template<typename TWType, typename TAType>
TWType wrap_ta_(TAType& t) {
    /// Quick default
    if(!t.is_initialized()) return TWType();

    /// Typedefs
    using field_type    = typename TWType::field_type;
    using ta_pimpl_type = buffer::detail_::TABufferPIMPL<field_type>;
    using buffer_type   = buffer::Buffer<field_type>;
    using shape_type    = Shape<field_type>;
    using pimpl_type    = detail_::TensorWrapperPIMPL<field_type>;

    /// Wrap input in a buffer
    auto pt      = std::make_unique<ta_pimpl_type>(std::forward<TAType>(t));
    auto pbuffer = std::make_unique<buffer_type>(std::move(pt));

    /// Get shape information
    auto extents       = pbuffer->make_extents();
    auto inner_extents = pbuffer->make_inner_extents();
    auto pshape        = std::make_unique<shape_type>(extents, inner_extents);

    // Make allocator
    auto palloc = allocator::ta_allocator<field_type>();

    // Move buffer, shape, and allocator into TensorWrapperPIMPL
    auto ppimpl = std::make_unique<pimpl_type>(
      std::move(pbuffer), std::move(pshape), std::move(palloc));

    // Finally make the tensor
    return TWType(std::move(ppimpl));
}

template<typename TWType, typename TAType>
TAType& unwrap_ta_(TWType& tw) {
    Conversion<TAType> converter;
    return converter.convert(tw.buffer());
}

/// A little bit cleaner typedef
using TSpArrayD   = TA::TSpArrayD;
using TSpArrayToD = TA::TSpArray<TA::Tensor<double>>;

ScalarTensorWrapper wrap_ta(TSpArrayD& ta) {
    return wrap_ta_<ScalarTensorWrapper, TSpArrayD>(ta);
}

TensorOfTensorsWrapper wrap_ta(TSpArrayToD& ta) {
    return wrap_ta_<TensorOfTensorsWrapper, TSpArrayToD>(ta);
}

TSpArrayD& unwrap_ta(ScalarTensorWrapper& tw) {
    return unwrap_ta_<ScalarTensorWrapper, TSpArrayD>(tw);
}

TSpArrayToD& unwrap_ta(TensorOfTensorsWrapper& tw) {
    return unwrap_ta_<TensorOfTensorsWrapper, TSpArrayToD>(tw);
}

} // namespace tensorwrapper::tensor::backends