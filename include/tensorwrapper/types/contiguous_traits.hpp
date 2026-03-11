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
#include <tensorwrapper/forward_declarations.hpp>
#include <tensorwrapper/types/buffer_traits.hpp>
#include <tensorwrapper/types/class_traits.hpp>
#include <tensorwrapper/types/shape_traits.hpp>

namespace tensorwrapper::types {

struct ContiguousTraitsCommon {
    using shape_type       = shape::Smooth;
    using const_shape_view = shape::SmoothView<const shape_type>;
};

template<>
struct ClassTraits<tensorwrapper::buffer::Contiguous>
  : public ClassTraits<buffer::Replicated>, public ContiguousTraitsCommon {};

template<>
struct ClassTraits<const tensorwrapper::buffer::Contiguous>
  : public ClassTraits<const buffer::Replicated>,
    public ContiguousTraitsCommon {};

} // namespace tensorwrapper::types
