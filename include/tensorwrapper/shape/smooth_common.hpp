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
#include <initializer_list>
#include <shape/shape_traits.hpp>
#include <vector>

namespace tensorwrapper::shape {

/** @brief Code factorization for Smooth/SmoothView.
 *
 *  @tparam Derived The class *this is implementing. Expected to be unqualified
 *                  Smooth or SmoothView.
 *
 *  To use this class the derived class must define:
 *  - `size_type extent(rank_type i) const` so that it returns the extent of
 *    mode i.
 */
template<typename Derived>
class SmoothCommon {
private:
    using traits_type = ShapeTraits<Derived>;

public:
    using rank_type     = typename traits_type::rank_type;
    using size_type     = typename traits_type::size_type;
    using slice_type    = typename traits_type::slice_type;
    using slice_il_type = std::initializer_list<size_type>;

    /** @brief Returns the extent of the @p i -th mode.
     *
     *  @param[in] i The mode the user wants the extent of. @p i must be in the
     *               range [0, rank()).
     *
     *  @return The extent of the requested mode.
     *
     *  @throw std::out_of_range if @p i is not in the range [0, range()).
     *                           Strong throw guarantee.
     */
    decltype(auto) extent(rank_type i) const {
        return derived().extent_impl(i);
    }

    /** @brief Slices a shape given two initializer lists.
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
    slice_type slice(slice_il_type first_elem, slice_il_type last_elem) const {
        return slice(first_elem.begin(), first_elem.end(), last_elem.begin(),
                     last_elem.end());
    }

    /** @brief Slices a shape given two containers.
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
    slice_type slice(ContainerType0&& first_elem,
                     ContainerType1&& last_elem) const {
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
     *
     */
    template<typename BeginItr, typename EndItr>
    slice_type slice(BeginItr first_elem_begin, BeginItr first_elem_end,
                     EndItr last_elem_begin, EndItr last_elem_end) const;

private:
    // Downcasts *this to mutable Derived object
    decltype(auto) derived() { return static_cast<Derived&>(*this); }

    // Downcasts *this to read-only Derived object
    decltype(auto) derived() const {
        return static_cast<const Derived&>(*this);
    }
};

// -----------------------------------------------------------------------------
// -- Out of line implementations
// -----------------------------------------------------------------------------

template<typename Derived>
template<typename BeginItr, typename EndItr>
auto SmoothCommon<Derived>::slice(BeginItr first_elem_begin,
                                  BeginItr first_elem_end,
                                  EndItr last_elem_begin,
                                  EndItr last_elem_end) const -> slice_type {
    std::vector<size_type> new_extents;

    auto first_done = [&]() { return first_elem_begin == first_elem_end; };
    auto last_done  = [&]() { return last_elem_begin == last_elem_end; };

    if(first_done() && last_done()) {
        // TODO: Assert rank 0
        return slice_type{};
    } else if(first_done() || last_done()) {
        throw std::runtime_error("Ranges were NOT equal");
    }

    for(; !first_done(); ++first_elem_begin, ++last_elem_begin) {
        if(last_done()) // Last ended before first
            throw std::runtime_error("Ranges were NOT equal");

        const auto ei = *last_elem_begin;
        const auto bi = *first_elem_begin;
        if(bi >= ei)
            throw std::runtime_error("begin index must be < end index");

        new_extents.push_back(ei - bi);
    }
    if(!last_done()) { throw std::runtime_error("Ranges were NOT equal"); }

    // TODO: assert rank == new_extents.size()
    return slice_type(new_extents.begin(), new_extents.end());
}

} // namespace tensorwrapper::shape
