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
#include <tensorwrapper/types/mdbuffer_traits.hpp>

namespace tensorwrapper::buffer {

/** @brief A multidimensional (MD) buffer.
 *
 *  This class is a dense multidimensional buffer of floating-point values.
 */
class MDBuffer {
private:
    using traits_type = types::ClassTraits<MDBuffer>;

public:
    /// Add types to public API
    ///@{
    using buffer_type   = typename traits_type::buffer_type;
    using pimpl_type    = typename traits_type::pimpl_type;
    using pimpl_pointer = typename traits_type::pimpl_pointer;
    using rank_type     = typename traits_type::rank_type;
    using shape_type    = typename traits_type::shape_type;
    ///@}

    MDBuffer() noexcept;

    template<typename T>
    MDBuffer(shape_type shape, std::vector<T> elements) {
        MDBuffer(std::move(shape), buffer_type(std::move(elements)));
    }

    MDBuffer(shape_type shape, buffer_type buffer);

    rank_type rank() const;

private:
    explicit MDBuffer(pimpl_pointer pimpl) noexcept;

    bool has_pimpl_() const noexcept;

    void assert_pimpl_() const;

    pimpl_type& pimpl_();
    const pimpl_type& pimpl_() const;

    pimpl_pointer m_pimpl_;
};

} // namespace tensorwrapper::buffer
