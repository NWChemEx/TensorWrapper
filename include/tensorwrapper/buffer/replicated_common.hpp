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
#include <tensorwrapper/concepts/has_begin_end.hpp>
#include <tensorwrapper/interfaces/sliceable.hpp>
#include <tensorwrapper/types/buffer_traits.hpp>

namespace tensorwrapper::buffer {

/** @brief Implements and defines common functionality for replicated buffers.
 *
 *  @tparam Derived The class *this is implementing. Expected to be unqualified
 *                  Replicated or ReplicatedView.
 *
 *  To use this class the derived class must define:
 *  - `const_element_reference get_elem_(index_vector index) const` so that it
 *    returns the element at the given index.
 *  - `void set_elem_(index_vector index, element_type value)` so that it sets
 *    the element at the given index to the given value.
 *
 *  This class is used to implement the common functionality for replicated
 *  buffers.
 */
template<typename Derived>
class ReplicatedCommon : public interfaces::Sliceable<Derived> {
private:
    using my_traits = types::ClassTraits<Derived>;

protected:
    using sliceable_base = interfaces::Sliceable<Derived>;

public:
    /// Pull in types from traits_type
    ///@{
    using element_type            = typename my_traits::element_type;
    using element_reference       = typename my_traits::element_reference;
    using const_element_reference = typename my_traits::const_element_reference;
    using size_type               = typename my_traits::size_type;
    using index_vector            = typename my_traits::index_vector;
    using slice_type              = typename my_traits::slice_type;
    using const_slice_type        = typename my_traits::const_slice_type;
    using slice_il_type           = typename my_traits::slice_il_type;
    ///@}

    /** @brief Returns the element at the given index.
     *
     *  @param[in] index The index of the element to return.
     *
     *  @return The element at the given index.
     */
    const_element_reference get_elem(index_vector index) const {
        return derived().get_elem_(index);
    }

    /** @brief Sets the element at the given index to the given value.
     *
     *  @param[in] index The index of the element to set.
     *  @param[in] value The value to set the element to.
     */
    void set_elem(index_vector index, element_type value) {
        derived().set_elem_(index, value);
    }

private:
    /// Access derived for CRTP
    Derived& derived() { return static_cast<Derived&>(*this); }

    /// Access derived for CRTP read-only
    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
};

} // namespace tensorwrapper::buffer
