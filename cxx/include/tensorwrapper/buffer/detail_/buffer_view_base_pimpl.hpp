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
#include <tensorwrapper/types/buffer_traits.hpp>
namespace tensorwrapper::buffer::detail_ {

/// PIMPL holding a non-owning pointer to a LayoutBase.
template<typename BufferBaseType>
class BufferViewBasePIMPL {
private:
    /// Type of the buffer base
    using buffer_base_type = BufferBaseType;
    using traits_type      = types::ClassTraits<buffer_base_type>;

public:
    using layout_type            = typename traits_type::layout_type;
    using layout_pointer         = typename traits_type::layout_pointer;
    using layout_reference       = typename traits_type::layout_reference;
    using const_layout_reference = typename traits_type::const_layout_reference;

    explicit BufferViewBasePIMPL(layout_pointer p) noexcept :
      m_layout_ptr_(p) {}

    layout_reference layout() { return *m_layout_ptr_; }
    const_layout_reference layout() const { return *m_layout_ptr_; }

    bool has_layout() const noexcept { return m_layout_ptr_ != nullptr; }

    auto clone() const { return std::make_unique<BufferViewBasePIMPL>(*this); }

private:
    layout_pointer m_layout_ptr_;
};
} // namespace tensorwrapper::buffer::detail_
