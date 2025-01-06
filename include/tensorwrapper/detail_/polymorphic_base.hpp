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
#include <string>
#include <tensorwrapper/detail_/unique_ptr_utilities.hpp>

namespace tensorwrapper::detail_ {

/** @brief Defines the API polymorphic utility methods should use.
 *
 *  @tparam BaseType the base class of the hierarchy being implemented.
 *
 *  The PolymorphicBase class is designed to make it easier to write a
 *  polymorphic object hierarchy by providing standardized APIs and default
 *  implementations for those functions.
 */
template<typename BaseType>
class PolymorphicBase {
public:
    /// Type *this is implementing
    using base_type = BaseType;

    /// Mutable reference to an object of type base_type
    using base_reference = base_type&;

    /// Read-only reference to an object of type base_type
    using const_base_reference = const base_type&;

    /// Mutable rvalue reference to an object of type base_type
    using base_rvalue = base_type&&;

    /// Pointer to an object of type base_type
    using base_pointer = std::unique_ptr<base_type>;

    /// Pointer to a read-only object of type base_type
    using const_base_pointer = std::unique_ptr<const base_type>;

    /// Type used for representing *this as a string
    using string_type = std::string;

    /// @brief Defaulted no-throw polymorphic dtor
    virtual ~PolymorphicBase() noexcept = default;

    /** @brief Creates a deep polymorphic copy of *this via the base class.
     *
     *  Calling the copy constructor of an object of type T is supposed to
     *  return a deep copy of the a T object. When T is polymorphic such a copy
     *  slices off the pieces of the object defined in classes which derive
     *  from T. Calling clone will ensure that the entire object is copied,
     *  including the pieces in derived classes.
     *
     *  Derived classes should override clone_ to implement this method.
     *
     *  @return A deep polymorphic copy of *this.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    base_pointer clone() const { return clone_(); }

    /** @brief Creates a deep polymorphic copy, but returns the pointer as a
     *         class other than base_type.
     *
     *  @tparam DerivedType What type of object should the resulting pointer
     *                      point to? This function will fail to compile if
     *                      base_type can not be cast (either implicitly or
     *                      explicitly) to a DerivedType object.
     *
     *  @return A deep copy of *this returned as a pointer to a @p DerivedType
     *          object.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    template<typename DerivedType>
    auto clone_as() const {
        auto pbase = clone();
        return detail_::static_pointer_cast<DerivedType>(pbase);
    }

    /** @brief Determines if *this and @p rhs are polymorphically equal.
     *
     *  Calling operator== on an object of type T is supposed to compare the
     *  state defined in class T as well as all state defined in parent
     * classes. If there is other classes which derived from T that possess
     * state, use of T::operator== will not consider such state in the
     * comparison. This method casts both *this and @p rhs to their most
     * derived class and then performs the value comparison to ensure that all
     * state is considered. If *this and @p rhs have different most derived
     * classes this comparison returns false.
     *
     *  Derived classes should override are_equal_ to implement this method.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this and @p rhs are polymorphically value equal and
     *          false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool are_equal(const_base_reference rhs) const noexcept {
        // Downcast *this so it can be passed to are_equal_
        const_base_reference plhs = static_cast<const_base_reference>(*this);

        // This line is necessary if are_equal_ is overriden in BaseType
        const PolymorphicBase& rhs_upcast = rhs;

        return are_equal_(rhs) && rhs_upcast.are_equal_(plhs);
    }

    /** @brief Determines if *this and @p rhs are polymorphically different.
     *
     *  Two objects are polymorphically different if they are not
     *  polymorphically value equal. See the documentation for are_equal for a
     *  definition of polymorphically value equal.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return False if *this and @p rhs are polymorphically value equal and
     *          true otherwise.
     *
     *  @throw None Strong throw guarantee.
     */
    bool are_different(const_base_reference rhs) const noexcept {
        return !are_equal(rhs);
    }

    /** @brief Returns a string representation of *this.
     *
     *  By default a polymorphic object is represented as `"{?}"`. Derived
     *  classes are encouraged to override `to_string_` to provide more
     *  helpful representations.
     *
     *  @note This method is meant primarily for logging/debugging and NOT for
     *        serialization or archival.
     *
     *  @return *this represented as a string.
     */
    auto to_string() const { return to_string_(); }

protected:
    /** @brief No-op default ctor
     *
     *  Users will not create objects of PolymorphicBase directly, which is why
     *  the default ctor is protected.
     *
     *  @throw None No throw guarantee.
     *
     */
    PolymorphicBase() noexcept = default;

    /** @brief Copy ctor.
     *
     *  This is a no-op because PolymorphicBase has no state. It is protected
     *  to avoid slicing.
     *
     *  @param[in] other The object to copy.
     *
     *  @throw None No throw guarantee.
     */
    PolymorphicBase(const PolymorphicBase& other) = default;

    /** @brief Derived classes should override this method to implement clone.
     *
     *  In most cases the derived class should implement this method by calling
     *  ``std::make_unique<DerivedType>(*this)` where `DerivedType` is the
     *  derived class's type.
     *
     *  @return A deep copy of the derived class.
     *
     *  @throw std::bad_alloc if there is a problem allocating the memory.
     *                        Strong throw guarantee.
     */
    virtual base_pointer clone_() const = 0;

    /** @brief Implements are_equal_ assuming the derived class implements
     *         operator==.
     *
     *  @tparam DerivedType The type of the derived class which is implementing
     *                      are_equal. This is an explicit template type
     *                      parameter and must be provided by the caller.
     *
     *  Polymorphic equality involves downcasting the objects to ensure they
     *  have the same derived type and then comparing the objects for value
     *  equality. Assuming @tparam DerivedType implements value equality via
     *  operator==, this procedure is the same for all classes and we factor it
     *  out into this method. Thus, to implement are_equal derived classes
     *  should just call this method.
     */
    template<typename DerivedType>
    bool are_equal_impl_(const_base_reference rhs) const noexcept {
        auto plhs = dynamic_cast<const DerivedType*>(this);
        auto prhs = dynamic_cast<const DerivedType*>(&rhs);
        if(plhs == nullptr || prhs == nullptr) return false;
        return (*plhs) == (*prhs);
    }

    /** @brief Derived classes should override this method to implement
     *         are_equal.
     *
     *   Each non-abstract derived class, `T`, should override this method so
     *   that it calls `are_equal_impl_<T>`. Assuming that `T::operator==` is
     *   defined for non-polymorphic value comparison, this will suffice for
     *   implementing are_equal.
     *
     *   @param[in] rhs The object to compare against.
     *
     *   @return True if *this and @p rhs are polymorphically value equal and
     *           false otherwise.
     */
    virtual bool are_equal_(const_base_reference rhs) const noexcept = 0;

    /// Should be overridden by the derived class to provide logging details.
    virtual string_type to_string_() const { return "{?}"; }
};

/// Implements printing via ostream for objects deriving from PolymorphicBase
template<typename T>
inline std::ostream& operator<<(std::ostream& os, const PolymorphicBase<T>& b) {
    return os << b.to_string();
}

} // namespace tensorwrapper::detail_
