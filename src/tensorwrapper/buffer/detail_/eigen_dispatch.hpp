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

#pragma once
#include <Eigen/CXX11/Tensor>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::buffer::detail_ {

constexpr std::size_t MaxEigenRank = 8;

template<typename FloatType, unsigned int Rank>
using eigen_tensor_type = eigen::Tensor<FloatType, Rank>;

template<typename FloatType, unsigned int Rank>
using eigen_tensor_map = eigen::TensorMap<eigen_tensor_type<FloatType, Rank>>;

template<typename FloatType, unsigned int Rank>
auto wrap_tensor(std::span<FloatType> s, const shape::Smooth& shape) {
    using tensor_type = eigen::Tensor<FloatType, Rank>;
    using map_type    = eigen::TensorMap<tensor_type>;

    if constexpr(Rank > MaxEigenRank) {
        static_assert(
          Rank <= MaxEigenRank,
          "Eigen tensors of rank > MaxEigenRank are not supported.");
    } else {
        if(shape.rank() == Rank) return variant_type(map_type(s));
    }
}

template<typename VisitorType, typename FloatType, unsigned int Rank,
         typename... Args>
auto eigen_dispatch_impl(VisitorType&& visitor,
                         eigen::TensorMap<eigen::Tensor<FloatType, Rank>>& A,
                         Args&&... args) {
    return visitor(A, std::forward<Args>(args)...);
}

template<typename VisitorType, typename FloatType, unsigned int Rank,
         typename... Args>
auto eigen_tensor_dispatch(std::span<FloatType> s, shape::Smooth shape,
                           Args&&... args) {
    using tensor_type = eigen::Tensor<FloatType, Rank>;
}

} // namespace tensorwrapper::buffer::detail_
