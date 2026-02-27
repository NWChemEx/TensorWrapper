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
#include <tensorwrapper/layout/layout_base.hpp>
#include <tensorwrapper/types/layout_traits.hpp>

namespace tensorwrapper::layout {

template<typename Derived>
class LayoutCommon : public LayoutBase {
private:
    /// Type of *this
    using my_type = LayoutCommon<Derived>;

    /// Type defining the types for *this
    using traits_type = types::ClassTraits<my_type>;

public:
    ///@{
    using slice_type     = typename traits_type::slice_type;
    using offset_il_type = typename traits_type::offset_il_type;
    ///@}

    /// Pull in base class's ctors
    using LayoutBase::LayoutBase;

    /** @brief Slices a layout given two initializer lists.
     *
     *  C++ doesn't allow templates to work with initializer lists, therefore
     *  we must provide a special overload for when the input containers are
     *  initializer lists. This method simply dispatches to the range-based
     *  method by calling begin()/end() on each initializer list. See the
     *  description of that method for more details.
     *
     *  @param[in] first_elem An initializer list containing the offsets of
     *                        the first element IN the slice such that
     *                        `first_elem[i]` is the offset along mode i.
     *  @param[in] last_elem An initializer list containing the offsets of
     *                        the first element NOT IN the slice such that
     *                        `last_elem[i]` is the offset along mode i.
     *
     *  @return The requested slice.
     *
     *  @throws ??? If the range-based method throws. Same throw guarantee.
     */
    slice_type slice(offset_il_type first_elem,
                     offset_il_type last_elem) const {
        return slice(first_elem.begin(), first_elem.end(), last_elem.begin(),
                     last_elem.end());
    }

    /** @brief Slices a layout given two containers.
     *
     *  @tparam ContainerType0 The type of first_elem. Assumed to have
     *                         begin()/end() methods.
     *  @tparam ContainerType1 The type of last_elem. Assumed to have
     *                         begin()/end() methods.
     *
     *  Element indices are usually stored in containers. This overload is a
     *  convenience method for calling begin()/end() on the containers before
     *  dispatching to the range-based overload. See the documentation for the
     *  range-based overload for more details.
     *
     *  @param[in] first_elem A container containing the offsets of
     *                        the first element IN the slice such that
     *                        `first_elem[i]` is the offset along mode i.
     *  @param[in] last_elem A container containing the offsets of
     *                        the first element NOT IN the slice such that
     *                        `last_elem[i]` is the offset along mode i.
     *
     *  @return The requested slice.
     *
     *  @throws ??? If the range-based method throws. Same throw guarantee.
     */
    template<typename ContainerType0, typename ContainerType1>
    slice_type slice(ContainerType0&& first_elem, ContainerType1&& last_elem) {
        return slice(first_elem.begin(), first_elem.end(), last_elem.begin(),
                     last_elem.end());
    }

    /** @brief Implements slicing given two ranges.
     *
     *  @tparam BeginItr The type of the iterators pointing to offsets in the
     *                   container holding the first element of the slice.
     *  @tparam EndItr The type of the iterators pointing to the offsets in
     *                 the container holding the first element NOT in the
     *                 slice.
     *
     *  All other slice functions dispatch to this method.
     *
     *  Slices are assumed to be contiguous, meaning we can uniquely specify
     *  the slice by providing the first element IN the slice and the first
     *  element NOT IN the slice.
     *
     *  Specifying an element of a rank @f$r@f$ tensor requires providing
     *  @f$r@f$ offsets (one for each mode). Generally speaking, this requires
     *  the offsets to be in a container. This method takes iterators to those
     *  containers such that the @f$r@f$ elements in the range
     *  [first_elem_begin, first_elem_end) are the offsets of first element IN
     *  the slice and [last_elem_begin, last_elem_end) are the offsets of the
     *  first element NOT IN the slice.
     *
     *  @note Both [first_elem_begin, first_elem_end) and
     *        [last_elem_begin, last_elem_end) being empty is allowed as long
     *        as *this is null or for a scalar. In these cases you will get back
     *        the only slice possible, which is the entire shape, i.e. a copy of
     *        *this.
     *
     *  @param[in] first_elem_begin An iterator to the offset along mode 0 for
     *             the first element in the slice.
     *  @param[in] first_elem_end An iterator pointing to just past the offset
     *             along mode "r-1" (r being the rank of *this) for the first
     *             element in the slice.
     *  @param[in] last_elem_begin An iterator to the offset along mode 0 for
     *             the first element NOT in the slice.
     *  @param[in] last_elem_end An iterator pointing to just past the offset
     *             along mode "r-1" (r being the rank of *this) for the first
     *             element NOT in the slice.
     *
     *  @return The requested slice.
     *
     *  @throw std::runtime_error if the range
     *            [first_elem_begin, first_elem_end) does not contain the same
     *            number of elements as [last_elem_begin, last_elem_end).
     *            Strong throw guarantee.
     * @throw std::runtime_error if the offsets in the range
     *            [first_elem_begin, first_elem_end) do not come before the
     *            offsets in [last_elem_begin, last_elem_end). Strong throw
     *            guarantee.
     * @throw std::runtime_error if [first_elem_begin, first_elem_end) and
     *                           [last_elem_begin, last_elem_end) contain the
     *                           same number of offsets, but that number is NOT
     *                           equal to the rank of *this. Strong throw
     *                           guarantee.
     *
     */
    template<typename BeginItr, typename EndItr>
    slice_type slice(BeginItr first_elem_begin, BeginItr first_elem_end,
                     EndItr last_elem_begin, EndItr last_elem_end) const;
};

template<typename Derived>
template<typename BeginItr, typename EndItr>
inline auto LayoutCommon<Derived>::slice(BeginItr first_elem_begin,
                                         BeginItr first_elem_end,
                                         EndItr last_elem_begin,
                                         EndItr last_elem_end) const
  -> slice_type {
    if(this->is_null()) return Derived{};
    auto new_shape = shape().as_smooth().slice(first_elem_begin, first_elem_end,
                                               last_elem_begin, last_elem_end);
    auto new_symmetry = symmetry().slice(first_elem_begin, first_elem_end,
                                         last_elem_begin, last_elem_end);
    auto new_sparsity = sparsity().slice(first_elem_begin, first_elem_end,
                                         last_elem_begin, last_elem_end);
    return slice_type{new_shape, new_symmetry, new_sparsity};
}

} // namespace tensorwrapper::layout
