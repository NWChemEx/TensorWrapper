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
#include <testing/eigen_buffers.hpp>

// This file contains some functions for creating TensorInput objects that span
// a number of use cases. This is meant to make it easier to test TensorWrapper
// with a number of different tensor setups.

namespace tensorwrapper::testing {

inline auto default_input() { return detail_::TensorInput{}; }

template<typename FloatType>
inline auto smooth_scalar_() {
    auto buffer = eigen_scalar<FloatType>();
    shape::Smooth shape{};
    return detail_::TensorInput(shape, std::move(buffer));
}

inline auto smooth_scalar_input() { return smooth_scalar_<double>(); }

/// 5 element vector such that element i is i
template<typename FloatType>
inline auto smooth_vector_() {
    auto buffer = eigen_vector<FloatType>();
    shape::Smooth shape{5};
    return detail_::TensorInput(shape, std::move(buffer));
}

inline auto smooth_vector_input() { return smooth_vector_<double>(); }

/// 5 element vector internally stored as a 5 by 1 matrix
inline auto smooth_vector_alt() {
    using buffer_type = buffer::Eigen<double, 2>;
    using data_type   = typename buffer_type::data_type;
    shape::Smooth shape{5};
    layout::Physical l(shape::Smooth{5, 1});
    data_type matrix(5, 1);
    for(std::size_t i = 0; i < 5; ++i) matrix(i, 0) = i;
    allocator::Eigen<double, 2> alloc(parallelzone::runtime::RuntimeView{});
    return detail_::TensorInput(shape, buffer_type(matrix, l, alloc));
}

template<typename FloatType>
inline auto smooth_matrix_() {
    auto buffer = eigen_matrix<FloatType>();
    shape::Smooth shape{2, 2};
    return detail_::TensorInput(shape, std::move(buffer));
}

inline auto smooth_matrix_input() { return smooth_matrix_<double>(); }

inline auto smooth_symmetric_matrix_input() {
    using buffer_type = buffer::Eigen<double, 2>;
    using data_type   = typename buffer_type::data_type;
    shape::Smooth shape{3, 3};
    layout::Physical l(shape);
    data_type matrix(3, 3);
    matrix(0, 0) = 1.0;
    matrix(0, 1) = 2.0;
    matrix(0, 2) = 3.0;
    matrix(1, 0) = 2.0;
    matrix(1, 1) = 4.0;
    matrix(1, 2) = 5.0;
    matrix(2, 0) = 3.0;
    matrix(2, 1) = 5.0;
    matrix(2, 2) = 6.0;
    symmetry::Permutation p01{0, 1};
    symmetry::Group g(p01);
    allocator::Eigen<double, 2> alloc(parallelzone::runtime::RuntimeView{});
    return detail_::TensorInput(shape, g, buffer_type(matrix, l, alloc));
}

template<typename FloatType>
inline auto smooth_tensor3_() {
    auto buffer = eigen_tensor3<FloatType>();
    shape::Smooth shape{2, 2, 2};
    return detail_::TensorInput(shape, std::move(buffer));
}

inline auto smooth_tensor3_input() { return smooth_tensor3_<double>(); }

inline auto smooth_tensor4_input() {
    using buffer_type = buffer::Eigen<double, 4>;
    using data_type   = typename buffer_type::data_type;
    shape::Smooth shape{2, 2, 2, 2};
    layout::Physical l(shape);
    data_type tensor(2, 2, 2, 2);
    tensor(0, 0, 0, 0) = 1.0;
    tensor(0, 0, 0, 1) = 2.0;
    tensor(0, 0, 1, 0) = 3.0;
    tensor(0, 0, 1, 1) = 4.0;
    tensor(0, 1, 0, 0) = 5.0;
    tensor(0, 1, 0, 1) = 6.0;
    tensor(0, 1, 1, 0) = 7.0;
    tensor(0, 1, 1, 1) = 8.0;
    tensor(1, 0, 0, 0) = 9.0;
    tensor(1, 0, 0, 1) = 10.0;
    tensor(1, 0, 1, 0) = 11.0;
    tensor(1, 0, 1, 1) = 12.0;
    tensor(1, 1, 0, 0) = 13.0;
    tensor(1, 1, 0, 1) = 14.0;
    tensor(1, 1, 1, 0) = 15.0;
    tensor(1, 1, 1, 1) = 16.0;
    allocator::Eigen<double, 4> alloc(parallelzone::runtime::RuntimeView{});
    return detail_::TensorInput(shape, buffer_type(tensor, l, alloc));
}

} // namespace tensorwrapper::testing
