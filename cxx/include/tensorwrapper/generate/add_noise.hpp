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
#include <cstdint>
#include <random>
#include <tensorwrapper/concepts/floating_point.hpp>
#include <tensorwrapper/tensor/tensor.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::generate {

/** @brief Adds clamped normal noise to each element of @p matrix.
 *
 *  Draws `delta ~ Normal(0, t)` and clamps to `[-t, t]` before adding to each
 *  element. When @p t is zero the input is copied unchanged.
 *
 *  For sigma uncertain types the result has standard deviation @p t. For sigma
 *  interval types the result has radius @p t about the perturbed center.
 *
 *  Explicit instantiations are provided only for types in
 *  @ref types::floating_point_types.
 *
 *  @tparam T Element type of the returned tensor.
 *
 *  @param[in] matrix The tensor to perturb.
 *  @param[in] t Non-negative noise scale (standard deviation and clamp bound).
 *  @param[in,out] gen Random number generator used for the normal draws.
 *
 *  @return A new tensor with the same shape as @p matrix.
 *
 *  @throw std::invalid_argument if @p t is negative.
 */
template<concepts::FloatingPoint T>
Tensor add_noise(const Tensor& matrix, double t, std::mt19937& gen);

/** @brief Overload of add_noise that creates its own RNG from @p seed.
 *
 *  @tparam T Element type of the returned tensor.
 *
 *  @param[in] matrix The tensor to perturb.
 *  @param[in] t Non-negative noise scale (standard deviation and clamp bound).
 *  @param[in] seed Seed for the internal random number generator. A value of
 *                  zero selects a non-deterministic seed.
 *
 *  @return A new tensor with the same shape as @p matrix.
 *
 *  @throw std::invalid_argument if @p t is negative.
 */
template<concepts::FloatingPoint T>
Tensor add_noise(const Tensor& matrix, double t, std::uint64_t seed = 42);

/** @brief Adds noise with element type `double`.
 *
 *  Equivalent to `add_noise<double>(matrix, t, gen)`.
 */
Tensor add_noise(const Tensor& matrix, double t, std::mt19937& gen);

/** @brief Adds noise with element type `double` using @p seed.
 *
 *  Equivalent to `add_noise<double>(matrix, t, seed)`.
 */
Tensor add_noise(const Tensor& matrix, double t, std::uint64_t seed = 42);

#define DECLARE_ADD_NOISE(TYPE)                                            \
    extern template Tensor add_noise<TYPE>(const Tensor& matrix, double t, \
                                           std::mt19937& gen);             \
    extern template Tensor add_noise<TYPE>(const Tensor& matrix, double t, \
                                           std::uint64_t seed);

TW_APPLY_FLOATING_POINT_TYPES(DECLARE_ADD_NOISE);

#undef DECLARE_ADD_NOISE

} // namespace tensorwrapper::generate
