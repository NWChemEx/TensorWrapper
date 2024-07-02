#pragma once
#include <deque>
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
class Group : public utilities::IndexableContainerBase<Group> {
private:
    /// Type of *this
    using my_type = Group;

    /// Type *this derives from
    using base_type = utilities::IndexableContainerBase<my_type>;

public:
    /// The base type of each object in *this
    using value_type = Operation;

    /// A mutable reference to a symmetry operation
    using reference = value_type::base_reference;

    /// A read-only reference to a symmetry operation
    using const_reference = value_type::const_base_reference;

    /// Unsigned integral type used for indexing and offsets
    using size_type = std::size_t;

    /** @brief Initializes *this to an empty group.
     *
     *  This ctor creates a group describing the symmetries of an empty set
     *  set. Such a set is also the symmetry group of a scalar.
     *
     *  @throw None No throw guarantee.
     */
    Group() noexcept = default;

    /** @brief Creates a Group from the provided symmetry operations.
     *
     *  @tparam Args The types of the remaining operations. Must be either an
     *               empty parameter pack or types that are implicitly
     *               convertible to const_reference.
     *
     *  This ctor accepts one or more symmetry operations and stores them in
     *  *this. Only unique operations are added, i.e., if @p op also appears in
     *  @p ops only one instance of @p op is added.
     *
     *  @param[in] op A symmetry operation to initialize the group with.
     *  @param[in] ops The remaining symmetry operations.
     *
     *  @throw std::bad_alloc if there is a problem allocating the initial
     *                        state. Strong throw guarantee.
     */
    template<typename... Args>
    explicit Group(const_reference op, Args&&... ops) :
      Group(std::forward<Args>(ops)...) {
        if(!count(op)) m_relations_.emplace_front(op.clone());
    }

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
    void swap(Group& other) noexcept { m_relations_.swap(other.m_relations_); }

    /** @brief Determines if *this is value equal to @p rhs.
     *
     *  Two Group objects are value equal if they contain the same number of
     *  operations and if each operation found in *this is also found in @p rhs.
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

private:
    /// Allow base class to access implementations
    friend base_type;

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
};

// -- Out of line implementations

inline bool Group::count(const_reference op) const noexcept {
    for(auto it = begin(); it != end(); ++it) {
        if(it->are_equal(op)) return true;
    }
    return false;
}

inline bool Group::operator==(const Group& rhs) const noexcept {
    if(size() != rhs.size()) return false;
    for(const auto& x : *this)
        if(!rhs.count(x)) return false;
    return true;
}

} // namespace tensorwrapper::symmetry
