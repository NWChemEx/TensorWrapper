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
#include <tensorwrapper/tensor/tensor.hpp>

namespace tensorwrapper::generate {

/** @brief Adds clamped normal noise to each element of @p matrix.
 *
 *  Draws `delta ~ Normal(0, t)` and clamps to `[-t, t]` before adding to each
 *  element. When @p t is zero the input is copied unchanged.
 *
 *  @param[in] matrix The tensor to perturb.
 *  @param[in] t Non-negative noise scale (standard deviation and clamp bound).
 *  @param[in,out] gen Random number generator used for the normal draws.
 *
 *  @return A new tensor with the same shape as @p matrix.
 *
 *  @throw std::invalid_argument if @p t is negative.
 */
Tensor add_noise(const Tensor& matrix, double t, std::mt19937& gen);

Tensor add_noise(const Tensor& matrix, double t, std::uint64_t seed = 42);

} // namespace tensorwrapper::generate
