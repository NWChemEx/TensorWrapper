/*
 * Copyright 2026 NWChemEx-Project
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

#pragma once
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::utilities {

template<typename BeginIterator, typename EndIterator>
Tensor make_tensor(std::vector<std::size_t> shape, BeginIterator begin,
                   EndIterator end) {
    shape::Smooth smooth_shape(shape.begin(), shape.end());
    std::vector data(begin, end);
    buffer::Contiguous buffer(data, smooth_shape);
    return Tensor(std::move(smooth_shape), std::move(buffer));
}

template<typename ContainerType>
Tensor make_tensor(std::initializer_list<std::size_t> shape,
                   ContainerType&& value) {
    return make_tensor(shape, value.begin(), value.end());
}

} // namespace tensorwrapper::utilities
