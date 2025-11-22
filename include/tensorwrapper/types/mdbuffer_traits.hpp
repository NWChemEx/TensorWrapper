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
#include <tensorwrapper/types/class_traits.hpp>
#include <tensorwrapper/types/shape_traits.hpp>
#include <wtf/wtf.hpp>

namespace tensorwrapper::types {

struct MDBufferTraitsCommon {
    using value_type        = wtf::fp::Float;
    using const_reference   = wtf::fp::FloatView<const value_type>;
    using buffer_type       = wtf::buffer::FloatBuffer;
    using const_buffer_view = wtf::buffer::BufferView<const value_type>;
    using shape_type        = shape::Smooth;
    using rank_type         = typename shape_type::rank_type;
    using pimpl_type        = tensorwrapper::buffer::detail_::MDBufferPIMPL;
    using pimpl_pointer     = std::unique_ptr<pimpl_type>;
};

template<>
struct ClassTraits<tensorwrapper::buffer::MDBuffer>
  : public MDBufferTraitsCommon {
    using reference = wtf::fp::FloatView<value_type>;

    using buffer_view       = wtf::buffer::BufferView<value_type>;
    using const_buffer_view = wtf::buffer::BufferView<const value_type>;
};

template<>
struct ClassTraits<const tensorwrapper::buffer::MDBuffer>
  : public MDBufferTraitsCommon {
    using reference   = wtf::fp::FloatView<const value_type>;
    using buffer_view = wtf::buffer::BufferView<const value_type>;
};

} // namespace tensorwrapper::types
