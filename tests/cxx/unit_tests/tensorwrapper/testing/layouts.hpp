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
#include "shapes.hpp"
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper::testing {

// -----------------------------------------------------------------------------
// -- Logical layouts
// -----------------------------------------------------------------------------

inline auto scalar_logical() {
    return tensorwrapper::layout::Logical(smooth_scalar());
}

inline auto vector_logical(std::size_t i = 10) {
    return tensorwrapper::layout::Logical(smooth_vector(i));
}

inline auto matrix_logical(std::size_t i = 10, std::size_t j = 10) {
    return tensorwrapper::layout::Logical(smooth_matrix(i, j));
}

inline auto tensor3_logical(std::size_t i = 10, std::size_t j = 10,
                            std::size_t k = 10) {
    return tensorwrapper::layout::Logical(smooth_tensor3(i, j, k));
}

inline auto tensor4_logical(std::size_t i = 10, std::size_t j = 10,
                            std::size_t k = 10, std::size_t l = 10) {
    return tensorwrapper::layout::Logical(smooth_tensor4(i, j, k, l));
}

// -----------------------------------------------------------------------------
// -- Physical layouts
// -----------------------------------------------------------------------------

inline auto scalar_physical() {
    return tensorwrapper::layout::Physical(smooth_scalar());
}

inline auto vector_physical(std::size_t i = 10) {
    return tensorwrapper::layout::Physical(smooth_vector(i));
}

inline auto matrix_physical(std::size_t i = 10, std::size_t j = 10) {
    return tensorwrapper::layout::Physical(smooth_matrix(i, j));
}

inline auto tensor3_physical(std::size_t i = 10, std::size_t j = 10,
                             std::size_t k = 10) {
    return tensorwrapper::layout::Physical(smooth_tensor3(i, j, k));
}

inline auto tensor4_physical(std::size_t i = 10, std::size_t j = 10,
                             std::size_t k = 10, std::size_t l = 10) {
    return tensorwrapper::layout::Physical(smooth_tensor4(i, j, k, l));
}

} // namespace tensorwrapper::testing