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
#include <deque>
#include <optional>
#include <tensorwrapper/detail_/dsl_base.hpp>
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/symmetry/operation.hpp>
#include <tensorwrapper/types/symmetry_traits.hpp>
#include <utilities/containers/indexable_container_base.hpp>

namespace tensorwrapper::symmetry {

/** @brief Container of the symmetry elements for a tensor.
 *
 *  Many tensors have elements which are related by symmetry. For example, a
 *  symmetric matrix is a matrix where the @f$(i,j)@f$-th element is the same
 *  as the @f$(j,i)@f$-th element. As the rank of the tensor increases, more
 *  symmetry relations are possible. The Group class models the set of symmetry
 *  operations which hold true for a given tensor.
 *
 *  @note At present the Group class does not actually assert that it is a
 *        group, e.g., if the user only provides the permutation (0, 1, 2) we
 *        mathematically know that the permutation (0, 2, 1) is also a symmetry
 *        operation because it is the inverse of (0, 1, 2).
 */
class Group : public utilities::IndexableContainerBase<Group>,
              public tensorwrapper::detail_::DSLBase<Group>,
              public tensorwrapper::detail_::PolymorphicBase<Group> {
private:
    /// Type of *this
    using my_type = Group;

    /// Type of the traits class defining the types for *this
    using traits_type = types::ClassTraits<my_type>;

    /// Type *this derives from to become container-like
    using container_type = utilities::IndexableContainerBase<my_type>;

    /// Type *this derives from to behave like other polymorphic objects
    using polymorphic_type = tensorwrapper::detail_::PolymorphicBase<Group>;

public:
    /// The base type of each object in *this
    using value_type = Operation;

    /// A mutable reference to a symmetry operation
    using reference = value_type::base_reference;

    /// A read-only reference to a symmetry operation
    using const_reference = value_type::const_base_reference;

    /// Pull in types from the traits class
    ///@{
    using size_type      = typename traits_type::size_type;
    using rank_type      = typename traits_type::rank_type;
    using slice_type     = typename traits_type::slice_type;
    using offset_il_type = typename traits_type::offset_il_type;
    ///@}

    // -------------------------------------------------------------------------
    // -- Ctors and assignment
    // -------------------------------------------------------------------------

    /** @brief Initializes *this to be the identity group of a scalar.
     *
     *  @throw None No throw guarantee
     */
    Group() noexcept = default;

    /** @brief Initializes *this as the identity group of a rank @p rank tensor.
     *
     *  This ctor creates a group representing the identity group of a rank
     *  @p rank tensor.
     *
     *  @param[in] rank The rank of the tensor the identity group describes.
     *
     *  @throw None No throw guarantee.
     */
    explicit Group(rank_type rank) noexcept : m_relations_{}, m_rank_(rank) {}

    /** @brief Creates a Group from the provided symmetry operations.
     *
     *  @tparam Args The types of the remaining operations. Must be either an
     *               empty parameter pack or types that are implicitly
     *               convertible to const_reference.
     *
     *  This ctor accepts one or more symmetry operations and stores them in
     *  *this. Only unique, non-identity operations are added, i.e., if @p op
     *  also appears in @p ops only one instance of @p op is added.
     *
     *  @param[in] op A symmetry operation to initialize the group with.
     *  @param[in] ops The remaining symmetry operations.
     *
     *  @throw std::runtime_error if @p op and @p ops do not all have the same
     *                            rank. Strong throw guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the initial
     *                        state. Strong throw guarantee.
     */
    template<typename... Args>
    explicit Group(const_reference op, Args&&... ops) :
      Group(std::forward<Args>(ops)...) {
        if(m_rank_ && rank() != op.rank())
            throw std::runtime_error("Ranks of operations are not consistent");
        if(!m_rank_) m_rank_.emplace(op.rank());
        if(!count(op) && !op.is_identity())
            m_relations_.emplace_front(op.clone());
    }

    /** @brief Deep copies the state of @p other.
     *
     *  The copy ctor will make deep (polymorphic) copies of each symmetry
     *  operation in @p other and then initialize *this with those copies.
     *
     *  @param[in] other The Group to deep copy.
     *
     *  @throw std::bad_alloc if there is a problem allocating the new state.
     *                        Strong throw guarantee.
     */
    Group(const Group& other) {
        for(const auto& x : other.m_relations_)
            m_relations_.push_back(x->clone());
        m_rank_ = other.m_rank_;
    }

    /** @brief Transfers the state of @p other in to *this.
     *
     *  @param[in,out] other The Group to take the state of. After this method
     *                       is called @p other will be in a valid, but
     *                       otherwise undefined state.
     *
     *  @throw None No throw guarantee.
     */
    Group(Group&& other) noexcept = default;

    /** @brief Replaces the state in *this with a copy of the state in @p rhs.
     *
     *  This method will release the state in *this and replace it with a deep
     *  copy of the state in @p rhs.
     *
     *  @param[in] rhs The Group to deep copy.
     *
     *  @return *this after replacing its state.
     *
     *  @throw std::bad_alloc if there is a problem allocating the new state.
     *                        Strong throw guarantee.
     */
    Group& operator=(const Group& rhs) {
        if(this != &rhs) Group(rhs).swap(*this);
        return *this;
    }

    /** @brief Replaces the state in *this with the state from @p rhs.
     *
     *  @param[in,out] rhs The Group to take the state of. After this method is
     *                     called @p rhs will be in a valid, but otherwise
     *                     undefined state.
     *
     *  @return *this after replacing its state with that of @p rhs.
     *
     *  @throw None No throw guarantee.
     */
    Group& operator=(Group&& rhs) noexcept = default;

    /** @brief Determines the number of times @p op appears in *this.
     *
     *  Since Group objects are set like this method determines whether or not
     *  @p op is contained in *this.
     *
     *  @param[in] op The operation we are looking for.
     *
     *  @return True if @p op appears in *this and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool count(const_reference op) const noexcept;

    /** @brief The rank of the tensor these symmetries describe.
     *
     *  This is not the rank of the group, but rather the rank of the tensor
     *  that the symmetries of the group describe.
     *
     *  @return The rank of the tensor described by the symmetries in *this.
     *
     *  @throw None No throw guarantee.
     */
    rank_type rank() const noexcept { return m_rank_.value_or(0); }

    /** @brief Slices a group given two initializer lists.
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

    /** @brief Slices a group given two containers.
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

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Exchanges the state of *this with that of @p other.
     *
     *  @param[in,out] other The object to exchange state with. After this
     *                       method is called @p other will contain the state
     *                       which was previously in *this.
     *
     *  @throw None No throw guarantee.
     */
    void swap(Group& other) noexcept {
        m_relations_.swap(other.m_relations_);
        m_rank_.swap(other.m_rank_);
    }

    /** @brief Determines if *this is value equal to @p rhs.
     *
     *  Two Group objects are value equal if they contain the same number of
     *  operations, if each operation found in *this is also found in @p rhs,
     *  and if the rank of the associated tensor is the same for *this as
     *  @p rhs.
     *
     *  @param[in] rhs The Group object we are comparing against.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const Group& rhs) const noexcept;

    /** @brief Determines if *this is different from @p rhs.
     *
     *  Two Group objects are defined as being different if they are not value
     *  equal. See operator== for the definition of value equal.
     *
     *  @param[in] rhs The object to compare against.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const Group& rhs) const noexcept { return !(*this == rhs); }

protected:
    typename polymorphic_type::base_pointer clone_() const override {
        return std::make_unique<Group>(*this);
    }

    bool are_equal_(const_base_reference rhs) const noexcept override {
        return (*this) == rhs;
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
    /// Allow base class to access implementations
    friend container_type;

    /// Base type common to all symmetry operations defining the API
    using value_pointer = value_type::base_pointer;

    /// Type of the container *this uses to hold the symmetry operations
    using relation_container_type = std::deque<value_pointer>;

    /// Implements mutable element access (though returned as read-only)
    const_reference at_(size_type i) noexcept { return *m_relations_[i]; }

    /// Implements read-only element access
    const_reference at_(size_type i) const noexcept { return *m_relations_[i]; }

    /// Implements size() with the number of explicit symmetry operations
    size_type size_() const noexcept { return m_relations_.size(); }

    /// The symmetry operations of *this
    relation_container_type m_relations_;

    /// The rank of the tensor these symmetries apply to
    std::optional<rank_type> m_rank_;
};

// -- Out of line implementations

inline bool Group::count(const_reference op) const noexcept {
    for(auto it = begin(); it != end(); ++it) {
        if(it->are_equal(op)) return true;
    }
    return false;
}

template<typename BeginItr, typename EndItr>
inline auto Group::slice(BeginItr first_elem_begin, BeginItr first_elem_end,
                         EndItr last_elem_begin, EndItr last_elem_end) const
  -> slice_type {
    if(size() != 0)
        throw std::runtime_error("Slicing of non-trivial symmetry NYI");

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

inline bool Group::operator==(const Group& rhs) const noexcept {
    if(rank() != rhs.rank()) return false;
    if(size() != rhs.size()) return false;
    for(const auto& x : *this)
        if(!rhs.count(x)) return false;
    return true;
}

} // namespace tensorwrapper::symmetry
