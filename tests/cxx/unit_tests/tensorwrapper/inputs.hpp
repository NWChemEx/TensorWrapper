/*
 * Copyright 2024 NWChemEx-Project
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
#include <tensorwrapper/tensorwrapper.hpp>

// This file contains some functions for creating TensorInput objects that span
// a number of use cases. This is meant to make it easier to test TensorWrapper
// with a number of different tensor setups.

namespace tensorwrapper::testing {

inline auto default_input() { return detail_::TensorInput{}; }

inline auto smooth_scalar() {
    using buffer_type = buffer::Eigen<double, 0>;
    using tensor_type = typename buffer_type::tensor_type;
    shape::Smooth shape{};
    layout::Physical l(shape);
    tensor_type scalar;
    scalar() = 42.0;
    return detail_::TensorInput(shape, buffer_type(scalar, l));
}

/// 5 element vector such that element i is i
inline auto smooth_vector() {
    using buffer_type = buffer::Eigen<double, 1>;
    using tensor_type = typename buffer_type::tensor_type;
    shape::Smooth shape{5};
    layout::Physical l(shape);
    tensor_type vector(5);
    for(std::size_t i = 0; i < 5; ++i) vector(i) = i;
    return detail_::TensorInput(shape, buffer_type(vector, l));
}

/// 5 element vector internally stored as a 5 by 1 matrix
inline auto smooth_vector_alt() {
    using buffer_type = buffer::Eigen<double, 2>;
    using tensor_type = typename buffer_type::tensor_type;
    shape::Smooth shape{5};
    layout::Physical l(shape::Smooth{5, 1});
    tensor_type matrix(5, 1);
    for(std::size_t i = 0; i < 5; ++i) matrix(i, 0) = i;
    return detail_::TensorInput(shape, buffer_type(matrix, l));
}

inline auto smooth_symmetric_matrix() {
    shape::Smooth shape{3, 3};
    symmetry::Permutation p01{0, 1};
    symmetry::Group g(p01);
    return detail_::TensorInput(shape, g);
}

} // namespace tensorwrapper::testing
