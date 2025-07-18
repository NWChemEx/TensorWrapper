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
    auto pbuffer = eigen_matrix<double>(5, 1);
    shape::Smooth shape{5};
    return detail_::TensorInput(shape, std::move(pbuffer));
}

template<typename FloatType>
inline auto smooth_matrix_(std::size_t n = 2, std::size_t m = 2) {
    auto buffer = eigen_matrix<FloatType>(n, m);
    shape::Smooth shape{n, m};
    return detail_::TensorInput(shape, std::move(buffer));
}

inline auto smooth_matrix_input() { return smooth_matrix_<double>(); }

inline auto smooth_symmetric_matrix_input() {
    auto pmatrix = eigen_matrix<double>(3, 3);
    pmatrix->set_elem({0, 0}, 1.0);
    pmatrix->set_elem({0, 1}, 2.0);
    pmatrix->set_elem({0, 2}, 3.0);
    pmatrix->set_elem({1, 0}, 2.0);
    pmatrix->set_elem({1, 1}, 4.0);
    pmatrix->set_elem({1, 2}, 5.0);
    pmatrix->set_elem({2, 0}, 3.0);
    pmatrix->set_elem({2, 1}, 5.0);
    pmatrix->set_elem({2, 2}, 6.0);
    shape::Smooth shape{3, 3};
    symmetry::Permutation p01{0, 1};
    symmetry::Group g(p01);

    return detail_::TensorInput(shape, g, std::move(pmatrix));
}

template<typename FloatType>
inline auto smooth_tensor3_() {
    auto buffer = eigen_tensor3<FloatType>();
    shape::Smooth shape{2, 2, 2};
    return detail_::TensorInput(shape, std::move(buffer));
}

inline auto smooth_tensor3_input() { return smooth_tensor3_<double>(); }

inline auto smooth_tensor4_input() {
    shape::Smooth shape{2, 2, 2, 2};
    auto pbuffer = eigen_tensor4<double>();
    return detail_::TensorInput(shape, std::move(pbuffer));
}

} // namespace tensorwrapper::testing
