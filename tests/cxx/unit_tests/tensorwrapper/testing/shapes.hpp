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

/** @file shapes.hpp
 *
 *  This file contains some already made shape objects to facilitate unit
 *  testing of TensorWrapper.
 */
#pragma once
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper::testing {

inline auto smooth_scalar() { return tensorwrapper::shape::Smooth{}; }

inline auto smooth_vector(std::size_t i = 10) {
    return tensorwrapper::shape::Smooth{i};
}

inline auto smooth_matrix(std::size_t i = 10, std::size_t j = 10) {
    return tensorwrapper::shape::Smooth{i, j};
}

inline auto smooth_tensor3(std::size_t i = 10, std::size_t j = 10,
                           std::size_t k = 10) {
    return tensorwrapper::shape::Smooth{i, j, k};
}

inline auto smooth_tensor4(std::size_t i = 10, std::size_t j = 10,
                           std::size_t k = 10, std::size_t l = 10) {
    return tensorwrapper::shape::Smooth{i, j, k, l};
}

} // namespace tensorwrapper::testing