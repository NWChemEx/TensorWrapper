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
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/types/class_traits.hpp>
#include <tensorwrapper/types/common_types.hpp>
#include <tensorwrapper/types/preserve_const.hpp>
#include <vector>
#include <wtf/wtf.hpp>

namespace tensorwrapper::types {

struct BufferBaseTraitsCommon : public CommonTypes {
    using layout_type            = layout::Physical;
    using const_layout_reference = const layout_type&;
    using const_layout_pointer   = const layout_type*;

    using buffer_base_type            = buffer::BufferBase;
    using const_buffer_base_pointer   = std::unique_ptr<const buffer_base_type>;
    using const_buffer_base_reference = const buffer_base_type&;
};

template<>
struct ClassTraits<buffer::BufferBase> : public BufferBaseTraitsCommon {
    using layout_reference      = layout_type&;
    using layout_pointer        = layout_type*;
    using buffer_base_reference = buffer_base_type&;
    using buffer_base_pointer   = std::unique_ptr<buffer_base_type>;
};

template<>
struct ClassTraits<const buffer::BufferBase> : public BufferBaseTraitsCommon {
    using layout_reference          = const layout_type&;
    using layout_pointer            = const layout_type*;
    using buffer_base_reference     = const buffer_base_type&;
    using buffer_base_pointer       = std::unique_ptr<const buffer_base_type>;
    using const_buffer_base_pointer = std::unique_ptr<const buffer_base_type>;
};

template<typename BufferBaseType>
struct ClassTraits<buffer::BufferViewBase<BufferBaseType>>
  : public ClassTraits<BufferBaseType> {};

template<>
struct ClassTraits<buffer::Local> : public ClassTraits<buffer::BufferBase> {};

template<>
struct ClassTraits<const buffer::Local>
  : public ClassTraits<const buffer::BufferBase> {};

template<typename LocalType>
struct ClassTraits<buffer::LocalView<LocalType>>
  : public ClassTraits<LocalType> {};

struct ReplicatedTraitsCommon {
    using element_type            = wtf::fp::Float;
    using const_element_reference = wtf::fp::FloatView<const element_type>;
    using buffer_type             = wtf::buffer::FloatBuffer;
    using const_buffer_view       = wtf::buffer::BufferView<const element_type>;
    using index_vector            = std::vector<types::CommonTypes::size_type>;
};

template<>
struct ClassTraits<buffer::Replicated> : public ReplicatedTraitsCommon,
                                         public ClassTraits<buffer::Local> {
    using element_reference = wtf::fp::FloatView<element_type>;
    using buffer_view       = wtf::buffer::BufferView<element_type>;
};

template<>
struct ClassTraits<const buffer::Replicated>
  : public ReplicatedTraitsCommon, public ClassTraits<const buffer::Local> {
    using element_reference = wtf::fp::FloatView<const element_type>;
    using buffer_view       = wtf::buffer::BufferView<const element_type>;
};

template<typename ReplicatedType>
struct ClassTraits<buffer::ReplicatedView<ReplicatedType>>
  : public ClassTraits<ReplicatedType> {};

} // namespace tensorwrapper::types
