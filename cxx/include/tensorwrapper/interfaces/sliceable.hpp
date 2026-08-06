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
#include <tensorwrapper/types/types.hpp>

namespace tensorwrapper::interfaces {

/** @brief Defines the interface for objects that can be sliced.
 *
 *  @tparam Derived The type of the derived class.
 *
 *  Assume we have a rank @f$r@f$ object. A slice of this object is a new object
 *  that is a rank @f$r@f$ object with a subset relationship with respect to the
 *  original object. Slices are contiguous in index space (for selecting
 *  arbitrary elements from the original object see masks), meaning the slice
 *  can be specified by providing two indices. The first index is the index of
 *  the first element IN the slice and the second index is the index of the
 *  first element NOT IN the slice. We term the first index in the slice is
 *  denoted @f$i_0@f$ and the first index not in the slice is denoted @f$i_N@f$
 *  where @f$N@f$ is the size of the slice (i.e. the number of elements in the
 *  slice).
 *
 *  TODO: Example of the range notation.
 *
 *  The Sliceable interface defines a number of overloads for the `slice` method
 *  these overloads are:
 *
 *  1. @f$i_0@f$ and @f$i_N@f$ are provided as @f$r@f$ element initializer
 *     lists. The returned slice is mutable.
 *  2. @f$i_0@f$ and @f$i_N@f$ are provided as @f$r@f$ element initalizer
 *     lists. The returned slice is read-only.
 *  3. @f$i_0@f$ and @f$i_N@f$ are provided as @f$r@f$ element containers that
 *     support `begin` and `end` iterators. The returned slice is mutable.
 *  4. @f$i_0@f$ and @f$i_N@f$ are provided as @f$r@f$ element containers that
 *     support `begin` and `end` iterators. The returned slice is read-only.
 *  5. @f$i_0@f$ and @f$i_N@f$ are provided as @f$r@f$ ranges of iterators. The
 *     returned slice is mutable.
 *  6. @f$i_0@f$ and @f$i_N@f$ are provided as @f$r@f$ ranges of iterators. The
 *     returned slice is read-only.
 *
 *  To use this interface the @p Derived class must:
 *
 *  1. Specialize `ClassTraits<Derived>` to provide a member type `slice_type`
 *      and `const_slice_type`.
 *  2. Implement the const and non-const versions of the `slice_` method.
 */
template<typename Derived>
class Sliceable {
private:
    using my_traits = types::ClassTraits<Derived>;

public:
    using index_vector     = typename my_traits::index_vector;
    using slice_type       = typename my_traits::slice_type;
    using const_slice_type = typename my_traits::const_slice_type;
    using slice_il_type    = typename my_traits::slice_il_type;

    /// Overload 1.
    slice_type slice(slice_il_type first_elem, slice_il_type last_elem) {
        return slice(first_elem.begin(), first_elem.end(), last_elem.begin(),
                     last_elem.end());
    }

    /// Overload 2.
    const_slice_type slice(slice_il_type first_elem,
                           slice_il_type last_elem) const {
        return slice_impl_(index_vector(first_elem.begin(), first_elem.end()),
                           index_vector(last_elem.begin(), last_elem.end()));
    }

    /// Overload 3.
    template<concepts::HasBeginEnd ContainerType0,
             concepts::HasBeginEnd ContainerType1>
    slice_type slice(ContainerType0&& first_elem, ContainerType1&& last_elem);

    /// Overload 4.
    template<concepts::HasBeginEnd ContainerType0,
             concepts::HasBeginEnd ContainerType1>
    const_slice_type slice(ContainerType0&& first_elem,
                           ContainerType1&& last_elem) const;

    /// Overload 5.
    template<std::forward_iterator BeginItr0, std::forward_iterator EndItr0,
             std::forward_iterator BeginItr1, std::forward_iterator EndItr1>
    slice_type slice(BeginItr0 first_elem_begin, EndItr0 first_elem_end,
                     BeginItr1 last_elem_begin, EndItr1 last_elem_end) {
        return slice_impl_(index_vector(first_elem_begin, first_elem_end),
                           index_vector(last_elem_begin, last_elem_end));
    }

    /// Overload 6.
    template<std::forward_iterator BeginItr0, std::forward_iterator EndItr0,
             std::forward_iterator BeginItr1, std::forward_iterator EndItr1>
    const_slice_type slice(BeginItr0 first_elem_begin, EndItr0 first_elem_end,
                           BeginItr1 last_elem_begin,
                           EndItr1 last_elem_end) const {
        return slice_impl_(index_vector(first_elem_begin, first_elem_end),
                           index_vector(last_elem_begin, last_elem_end));
    }

private:
    slice_type slice_impl_(index_vector first_elem, index_vector last_elem) {
        return derived().slice_(first_elem, last_elem);
    }

    const_slice_type slice_impl_(index_vector first_elem,
                                 index_vector last_elem) const {
        return derived().slice_(first_elem, last_elem);
    }

    Derived& derived() { return static_cast<Derived&>(*this); }

    const Derived& derived() const {
        return static_cast<const Derived&>(*this);
    }
};

// -----------------------------------------------------------------------------
// -- Out of line implementations
// -----------------------------------------------------------------------------

template<typename Derived>
template<concepts::HasBeginEnd ContainerType0,
         concepts::HasBeginEnd ContainerType1>
auto Sliceable<Derived>::slice(ContainerType0&& first_elem,
                               ContainerType1&& last_elem) -> slice_type {
    if constexpr(std::is_same_v<std::decay_t<ContainerType0>, index_vector> &&
                 std::is_same_v<std::decay_t<ContainerType1>, index_vector>) {
        return slice_impl_(first_elem, last_elem);
    } else {
        return slice_impl_(index_vector(first_elem.begin(), first_elem.end()),
                           index_vector(last_elem.begin(), last_elem.end()));
    }
}

template<typename Derived>
template<concepts::HasBeginEnd ContainerType0,
         concepts::HasBeginEnd ContainerType1>
auto Sliceable<Derived>::slice(ContainerType0&& first_elem,
                               ContainerType1&& last_elem) const
  -> const_slice_type {
    if constexpr(std::is_same_v<std::decay_t<ContainerType0>, index_vector> &&
                 std::is_same_v<std::decay_t<ContainerType1>, index_vector>) {
        return slice_impl_(first_elem, last_elem);
    } else {
        return slice_impl_(index_vector(first_elem.begin(), first_elem.end()),
                           index_vector(last_elem.begin(), last_elem.end()));
    }
}

} // namespace tensorwrapper::interfaces
