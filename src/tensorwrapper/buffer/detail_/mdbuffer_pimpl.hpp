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
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/types/mdbuffer_traits.hpp>

namespace tensorwrapper::buffer::detail_ {

class MDBufferPIMPL {
public:
    using parent_type = tensorwrapper::buffer::MDBuffer;
    using traits_type = tensorwrapper::types::ClassTraits<parent_type>;

    /// Add types to public API
    ///@{
    using value_type  = typename traits_type::value_type;
    using rank_type   = typename traits_type::rank_type;
    using buffer_type = typename traits_type::buffer_type;
    using shape_type  = typename traits_type::shape_type;
    ///@}

    MDBufferPIMPL(shape_type shape, buffer_type buffer) noexcept :
      m_shape_(std::move(shape)), m_buffer_(std::move(buffer)) {}

    auto& shape() noexcept { return m_shape_; }

    const auto& shape() const noexcept { return m_shape_; }

    auto& buffer() noexcept { return m_buffer_; }

    const auto& buffer() const noexcept { return m_buffer_; }

private:
    shape_type m_shape_;

    buffer_type m_buffer_;
};

} // namespace tensorwrapper::buffer::detail_
