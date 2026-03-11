/*
 * Copyright 2024 NWChemEx-Project
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
#include <tensorwrapper/types/buffer_traits.hpp>
#include <type_traits>

namespace tensorwrapper::buffer::detail_ {

/** @brief Implements the API for all ReplicatedView PIMPLs.
 *
 *  @tparam ReplicatedType The type *this will be a view of.
 */
template<typename ReplicatedType>
class ReplicatedViewPIMPL {
private:
    using traits_type = types::ClassTraits<ReplicatedType>;

public:
    /// Pull in types from traits_type
    ///@{
    using layout_reference       = typename traits_type::layout_reference;
    using const_layout_reference = typename traits_type::const_layout_reference;
    using layout_type            = typename traits_type::layout_type;
    using size_type              = typename traits_type::size_type;
    using element_type           = typename traits_type::element_type;
    using const_element_reference =
      typename traits_type::const_element_reference;
    using index_vector = typename traits_type::index_vector;
    ///@}

    /// Type of a pointer to the PIMPL
    using pimpl_pointer = std::unique_ptr<ReplicatedViewPIMPL>;

    /// No-throw dtor.
    virtual ~ReplicatedViewPIMPL() noexcept = default;

    pimpl_pointer clone() const { return clone_(); }

    layout_reference layout() { return layout_(); }

    const_layout_reference layout() const { return layout_(); }

    const_element_reference get_elem(const index_vector& slice_index) const {
        return get_elem_(slice_index);
    }

    template<typename T = ReplicatedType>
    std::enable_if_t<!std::is_const_v<T>> set_elem(
      const index_vector& slice_index, element_type value) {
        set_elem_(slice_index, std::move(value));
    }

protected:
    virtual layout_reference layout_() = 0;

    virtual const_layout_reference layout_() const = 0;

    virtual pimpl_pointer clone_() const = 0;

    /// Derived class should implement to be consistent with get_elem
    virtual const_element_reference get_elem_(
      const index_vector& slice_index) const = 0;

    /// Derived class should implement to be consistent with set_elem
    virtual void set_elem_(const index_vector& slice_index,
                           element_type value) = 0;
};

} // namespace tensorwrapper::buffer::detail_
