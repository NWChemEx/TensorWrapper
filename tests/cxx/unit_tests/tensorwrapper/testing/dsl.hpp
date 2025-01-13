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
#include "layouts.hpp"
#include "shapes.hpp"
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper::testing {

/// Types that participate in the DSL
using dsl_types =
  std::tuple<tensorwrapper::shape::Smooth, tensorwrapper::symmetry::Group,
             tensorwrapper::sparsity::Pattern, tensorwrapper::layout::Logical,
             tensorwrapper::layout::Physical, tensorwrapper::Tensor>;

inline auto scalar_values() {
    return dsl_types{smooth_scalar(),
                     tensorwrapper::symmetry::Group(0),
                     tensorwrapper::sparsity::Pattern(0),
                     scalar_logical(),
                     scalar_physical(),
                     Tensor(42.0)};
}

inline auto vector_values() {
    return dsl_types{smooth_vector(),
                     tensorwrapper::symmetry::Group(1),
                     tensorwrapper::sparsity::Pattern(1),
                     vector_logical(),
                     vector_physical(),
                     Tensor{1.0, 2.0, 3.0}};
}

inline auto matrix_values() {
    return dsl_types{smooth_matrix(),
                     tensorwrapper::symmetry::Group(2),
                     tensorwrapper::sparsity::Pattern(2),
                     matrix_logical(),
                     matrix_physical(),
                     Tensor{{1.0, 2.0}, {3.0, 4.0}}};
}

} // namespace tensorwrapper::testing