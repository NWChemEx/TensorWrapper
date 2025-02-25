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
#include <memory>
#include <tensorwrapper/buffer/buffer_fwd.hpp>
#include <tensorwrapper/types/class_traits.hpp>

namespace tensorwrapper::types {

template<>
struct ClassTraits<buffer::BufferBase> {
    /// Type all buffers inherit from
    using buffer_base_type = buffer::BufferBase;

    /// Type of a mutable reference to a buffer_base_type object
    using buffer_base_reference = buffer_base_type&;

    /// Type of a read-only reference to a buffer_base_type object
    using const_buffer_base_reference = const buffer_base_type&;

    /// Type of a unique_ptr to a mutable buffer_base_type
    using buffer_base_pointer = std::unique_ptr<buffer_base_type>;

    /// Type of a unique_ptr to a mutable buffer_base_type
    using const_buffer_base_pointer = std::unique_ptr<const buffer_base_type>;
};

} // namespace tensorwrapper::types