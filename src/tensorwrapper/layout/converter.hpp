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
#include <tensorwrapper/layout/logical.hpp>
#include <tensorwrapper/layout/physical.hpp>

namespace tensorwrapper::layout {

/** @brief Converts a logical layout into a physical layout. */
class Converter {
public:
    using logical_type            = Logical;
    using const_logical_reference = const logical_type&;
    using physical_type           = Physical;
    using physical_pointer        = std::unique_ptr<physical_type>;

    physical_pointer convert(const_logical_reference logical) {
        return std::make_unique<physical_type>(
          logical.shape(), logical.symmetry(), logical.sparsity());
    }
};

} // namespace tensorwrapper::layout