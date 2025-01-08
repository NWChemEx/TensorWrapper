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
#include <tensorwrapper/detail_/dsl_base.hpp>
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/symmetry/operation.hpp>
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

    /// Unsigned integral type used for indexing and offsets
    using size_type = std::size_t;

    /// Type used for mode indices
    using mode_index_type = typename value_type::mode_index_type;

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
    explicit Group(mode_index_type rank) noexcept :
      m_relations_{}, m_rank_(rank) {}

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
    mode_index_type rank() const noexcept { return m_rank_.value_or(0); }

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
    std::optional<mode_index_type> m_rank_;
};

// -- Out of line implementations

inline bool Group::count(const_reference op) const noexcept {
    for(auto it = begin(); it != end(); ++it) {
        if(it->are_equal(op)) return true;
    }
    return false;
}

inline bool Group::operator==(const Group& rhs) const noexcept {
    if(rank() != rhs.rank()) return false;
    if(size() != rhs.size()) return false;
    for(const auto& x : *this)
        if(!rhs.count(x)) return false;
    return true;
}

} // namespace tensorwrapper::symmetry
