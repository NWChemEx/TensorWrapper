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
#include <tensorwrapper/generate/generate_utils.hpp>
#include <tensorwrapper/tensor/tensor.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::generate {

/** @brief Generates a sorted rank-1 tensor of eigenvalues from @p spec.
 *
 *  The spectrum spans
 *  @f$[\texttt{spec.min\_eigenvalue},
 *  \texttt{spec.min\_eigenvalue} \times \texttt{spec.condition\_number}]@f$
 *  using the spacing strategy given by @p spec.spacing.
 *
 *  Explicit instantiations are provided only for types in
 *  @ref types::floating_point_types.
 *
 *  @tparam T Element type of the returned tensor.
 *
 *  @param[in] spec Parameters controlling the eigenvalue distribution.
 *  @param[in,out] gen Random number generator used when @p spec.spacing
 * requires random draws.
 *
 *  @return A rank-1 tensor of length @p spec.n containing the eigenvalues in
 *          ascending order.
 *
 *  @throw std::invalid_argument if @p spec.n is outside `[1, kMaxMatrixDim]` or
 *                               if @p spec.spacing is invalid.
 */
template<concepts::FloatingPoint T>
Tensor generate_eigenvalues(const SymmetricMatrixSpec& spec, std::mt19937& gen);

/** @brief Generates eigenvalues with element type `double`.
 *
 *  Equivalent to `generate_eigenvalues<double>(spec, gen)`.
 *
 *  @param[in] spec Parameters controlling the eigenvalue distribution.
 *  @param[in,out] gen Random number generator used when @p spec.spacing
 * requires random draws.
 *
 *  @return A rank-1 tensor of length @p spec.n containing the eigenvalues in
 *          ascending order.
 */
Tensor generate_eigenvalues(const SymmetricMatrixSpec& spec, std::mt19937& gen);

#define DECLARE_GENERATE_EIGENVALUES(TYPE)             \
    extern template Tensor generate_eigenvalues<TYPE>( \
      const SymmetricMatrixSpec& spec, std::mt19937& gen);

TW_APPLY_FLOATING_POINT_TYPES(DECLARE_GENERATE_EIGENVALUES);

#undef DECLARE_GENERATE_EIGENVALUES

} // namespace tensorwrapper::generate
