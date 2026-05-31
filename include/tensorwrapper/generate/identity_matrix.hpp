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
#include <cstddef>
#include <tensorwrapper/concepts/floating_point.hpp>
#include <tensorwrapper/generate/generate_utils.hpp>
#include <tensorwrapper/utilities/diagonal_matrix.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <vector>

namespace tensorwrapper::generate {

/** @brief Creates an @p n x @p n identity matrix.
 *
 *  @param[in] n Matrix dimension. Must be in `[1, kMaxMatrixDim]`.
 *
 *  @return A rank-2 tensor with ones on the diagonal and zeros elsewhere.
 *
 *  @throw std::invalid_argument if @p n is outside the allowed range.
 */
template<concepts::FloatingPoint T>
Tensor identity_matrix(std::size_t n) {
    require_valid_n(n);
    std::vector<T> values(n, 1.0);
    return utilities::diagonal_matrix(utilities::make_tensor({n}, values));
}

/** @brief Creates an identity matrix with element type `double`.
 *
 *  Equivalent to `identity_matrix<double>(n)`.
 *
 *  @param[in] n Matrix dimension. Must be in `[1, kMaxMatrixDim]`.
 *
 *  @return A rank-2 tensor with ones on the diagonal and zeros elsewhere.
 */
inline Tensor identity_matrix(std::size_t n) {
    return identity_matrix<double>(n);
}

} // namespace tensorwrapper::generate
