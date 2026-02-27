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

namespace tensorwrapper::types {

/** @brief Types used throughout the entire TensorWrapper library. */
struct CommonTypes {
    /// Type used for indexing and offsets along modes
    using size_type = std::size_t;

    /// Type used for describing the rank of a tensor and selecting a mode.
    using rank_type = unsigned short;
};

} // namespace tensorwrapper::types
