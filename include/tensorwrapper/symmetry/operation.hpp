#pragma once
#include <memory>

namespace tensorwrapper::symmetry {

/** @brief Common API for classes describing a symmetry operation.
 *
 *  The Group class interacts with the elements of the group through a common
 *  API. This class defines that API. The Operation class itself models a
 *  transformation which when applied to a tensor leaves the tensor unchanged.
 */
class Operation {
public:
    /// Common base class for all symmetry operations
    using base_type = Operation;

    /// Type of a reference to an operation's base class
    using base_reference = base_type&;

    /// Type of a read-only reference to an operation's base class
    using const_base_reference = const base_type&;

    /// Type of a pointer to a symmetry Operation's base class
    using base_pointer = std::unique_ptr<base_type>;

    /// Type used to index tensor modes
    using mode_index_type = unsigned short;

    // -------------------------------------------------------------------------
    // -- Ctors, assignment, and dtor
    // -------------------------------------------------------------------------

    /// Defaulted no-throw dtor
    virtual ~Operation() noexcept = default;

    /** @brief Polymorphic copy constructor.
     *
     *  Derived classes implement this method by overriding clone_
     *
     *  @return A deep copy of the derived class, returned as a pointer to
     *          *this.
     *
     *  @throw std::bad_alloc if there is a problem allocating the new state.
     *                        Strong throw guarantee.
     */
    base_pointer clone() const { return clone_(); }

    // -------------------------------------------------------------------------
    // - Properties
    // -------------------------------------------------------------------------

    bool is_identity() const noexcept { return is_identity_(); }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Determines if two Operation objects are polymorphically value
     *         equal.
     *
     *  Two Operation objects @f$a@f$ and @f$b@f$ are polymorphically value
     *  equal if the most derived class of @f$a@f$, @f$A@f$ is the same as the
     *  most derived class of @f$b@f$ and if when compared as objects of typ
     *  @f$A@f$ @f$a@f$ anb @f$b@f$ are value equal.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this is polymorphically value equal to @p rhs and false
     *          otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool are_equal(const_base_reference rhs) const noexcept {
        return are_equal_(rhs) && rhs.are_equal_(*this);
    }

protected:
    /** @brief Derived class should call to implement are_equal_
     *
     *  @tparam DerivedType The class we are implementing are_equal for.
     *
     *  Assuming the derived class implements operator== for non-polymorphic
     *  comparison, then are_equal can be implemented generically given the
     *  type of the derived class. This method is that generic implementation
     *  and should be called by the derived class.
     *
     *  @param[in] rhs The object to polymorphically compare to *this.
     *
     *  @return True if @p other compares value equal to *this and false
     *          otherwise.
     *
     *  @throw None No throw guarantee.
     */
    template<typename DerivedType>
    bool are_equal_impl_(const_base_reference rhs) const noexcept {
        auto pthis = dynamic_cast<const DerivedType*>(this);
        auto prhs  = dynamic_cast<const DerivedType*>(&rhs);
        if(pthis == nullptr || prhs == nullptr) return false;
        return (*pthis) == (*prhs);
    }

    /// Derived class should overwrite to implement clone()
    virtual base_pointer clone_() const = 0;

    /// Derived class should overwrite to implement is_identity
    virtual bool is_identity_() const noexcept = 0;

    /// Derived class should overwrite to implement are_equal()
    virtual bool are_equal_(const_base_reference rhs) const noexcept = 0;
};

} // namespace tensorwrapper::symmetry
