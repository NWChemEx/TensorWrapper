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
#include <random>
#include <tensorwrapper/concepts/floating_point.hpp>
#include <tensorwrapper/tensor/tensor.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::generate {

/** @brief Creates a random @p n x @p n orthogonal matrix.
 *
 *  Draws entries from a standard normal distribution and applies Householder QR
 *  factorization to obtain an orthogonal matrix @f$Q@f$.
 *
 *  Explicit instantiations are provided only for types in
 *  @ref types::floating_point_types.
 *
 *  @tparam T Element type of the returned tensor.
 *
 *  @param[in] n Matrix dimension. Must be in `[1, kMaxMatrixDim]`.
 *  @param[in,out] gen Random number generator used for the normal draws.
 *
 *  @return A rank-2 tensor whose columns form an orthonormal basis.
 *
 *  @throw std::invalid_argument if @p n is outside the allowed range.
 */
template<concepts::FloatingPoint T>
Tensor random_orthogonal_matrix(std::size_t n, std::mt19937& gen);

/** @brief Creates a random orthogonal matrix with element type `double`.
 *
 *  Equivalent to `random_orthogonal_matrix<double>(n, gen)`.
 *
 *  @param[in] n Matrix dimension. Must be in `[1, kMaxMatrixDim]`.
 *  @param[in,out] gen Random number generator used for the normal draws.
 *
 *  @return A rank-2 tensor whose columns form an orthonormal basis.
 */
Tensor random_orthogonal_matrix(std::size_t n, std::mt19937& gen);

#define DECLARE_RANDOM_ORTHOGONAL_MATRIX(TYPE)                           \
    extern template Tensor random_orthogonal_matrix<TYPE>(std::size_t n, \
                                                          std::mt19937& gen);

TW_APPLY_FLOATING_POINT_TYPES(DECLARE_RANDOM_ORTHOGONAL_MATRIX);

#undef DECLARE_RANDOM_ORTHOGONAL_MATRIX

} // namespace tensorwrapper::generate
