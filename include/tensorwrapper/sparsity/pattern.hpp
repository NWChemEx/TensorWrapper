/*
 * Copyright 2024 NWChemEx Community
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
#include <tensorwrapper/detail_/dsl_base.hpp>
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/types/sparsity_traits.hpp>

namespace tensorwrapper::sparsity {

/** @brief Base class for objects describing the sparsity of a tensor. */
class Pattern : public tensorwrapper::detail_::DSLBase<Pattern>,
                public tensorwrapper::detail_::PolymorphicBase<Pattern> {
private:
    /// Type defining the polymorphic API of *this
    using polymorphic_base = tensorwrapper::detail_::PolymorphicBase<Pattern>;

    /// Type defining the types for *this
    using traits_type = types::ClassTraits<Pattern>;

public:
    /// Add types to public API
    ///@{
    using size_type      = traits_type::size_type;
    using rank_type      = traits_type::rank_type;
    using offset_il_type = traits_type::offset_il_type;
    using slice_type     = traits_type::slice_type;
    ///@}

    /** @brief Creates a pattern for a rank @p rank tensor.
     *
     *  This constructor creates a sparsity pattern for a dense tensor with
     *  @p rank modes.
     *
     *  @param[in] rank The number of modes in the associated tensor.
     *
     *  @throw None No throw guarantee.
     */
    Pattern(rank_type rank = 0) noexcept : m_rank_(rank) {}

    /** @brief Provides the rank of the tensor *this assumes.
     *
     *  @return The rank of the tensor *this describes.
     *
     *  @throw None No throw guarantee.
     */
    rank_type rank() const noexcept { return m_rank_; }

    /** @brief Slices a sparsity pattern given two initializer lists.
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

    /** @brief Slices a sparsity pattern given two containers.
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

    /** @brief Determines if *this and @p rhs describe the same sparsity
     *         pattern.
     *
     *  At present the sparsity component only tracks the rank of the tensor so
     *  two Patterns are value equal if they describe tensors with the same
     *  rank.
     *
     *  @param[in] rhs The object to compare against.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const Pattern& rhs) const noexcept {
        return rank() == rhs.rank();
    }

    /** @brief Is *this different from @p rhs?
     *
     *  This class defines "different" as not value equal. See the description
     *  of operator== for the definition of value equal.
     *
     *  @param[in] rhs The object to compare against
     *
     *  @return False if *this and @p rhs are value equal and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const Pattern& rhs) const noexcept {
        return !((*this) == rhs);
    }

protected:
    /// Implements clone by calling copy constructor
    typename polymorphic_base::base_pointer clone_() const override {
        return std::make_unique<Pattern>(*this);
    }

    /// Implements are_equal by calling implementation provided by the base
    bool are_equal_(const_base_reference rhs) const noexcept override {
        return are_equal_impl_<Pattern>(rhs);
    }

    /// Implements addition_assignment via permute_assignment
    dsl_reference addition_assignment_(label_type this_labels,
                                       const_labeled_reference lhs,
                                       const_labeled_reference rhs) override;

    /// Implements subtraction_assignment via permute_assignment
    dsl_reference subtraction_assignment_(label_type this_labels,
                                          const_labeled_reference lhs,
                                          const_labeled_reference rhs) override;

    /// Implements multiplication_assignment via permute_assignment
    dsl_reference multiplication_assignment_(
      label_type this_labels, const_labeled_reference lhs,
      const_labeled_reference rhs) override;

    /// Implements permute_assignment by permuting the extents in @p rhs.
    dsl_reference permute_assignment_(label_type this_labels,
                                      const_labeled_reference rhs) override;

private:
    /// The rank of the tensor associated with *this
    rank_type m_rank_;
};

template<typename BeginItr, typename EndItr>
inline auto Pattern::slice(BeginItr first_elem_begin, BeginItr first_elem_end,
                           EndItr last_elem_begin, EndItr last_elem_end) const
  -> slice_type {
    auto first_done = [&]() { return first_elem_begin == first_elem_end; };
    auto last_done  = [&]() { return last_elem_begin == last_elem_end; };

    if(first_done() && last_done()) {
        if(rank() == 0) return slice_type{};
        throw std::runtime_error("Offset ranks does not match tensor rank");
    } else if(first_done() || last_done()) {
        throw std::runtime_error("Offsets do not have the same rank");
    }

    rank_type counter = 0;
    for(; !first_done(); ++first_elem_begin, ++last_elem_begin) {
        if(last_done())
            throw std::runtime_error("Offsets do not have the same rank.");

        auto fi = *first_elem_begin;
        auto li = *last_elem_begin;
        if(li <= fi)
            throw std::runtime_error("First element in slice must be strictly "
                                     "less than last element.");

        ++counter;
    }
    if(!last_done())
        throw std::runtime_error("Offsets do not have the same rank");
    if(counter != rank())
        throw std::runtime_error("Offset ranks do not match tensor rank");

    return slice_type(rank());
}

} // namespace tensorwrapper::sparsity
