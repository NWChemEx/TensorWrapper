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
#include <cstddef>
#include <memory>

namespace tensorwrapper::shape {

/** @brief Code factorization for the various types of shapes.
 *
 *  Full design details:
 *  https://nwchemex.github.io/TensorWrapper/developer/design/shape.html
 *
 *  All shapes posses a concept of:
 *  - Total rank
 *  - Total number of elements
 *
 *  To respectively implement these features, classes derived from *this are
 *  expected to implement:
 *  - get_rank_()
 *  - get_size_()
 */
class ShapeBase {
public:
    /// Type all shapes inherit from
    using shape_base = ShapeBase;

    /// Type of a pointer to the base of a shape object
    using base_pointer = std::unique_ptr<shape_base>;

    /// Type used to hold the rank of a tensor
    using rank_type = unsigned short;

    /// Type used to specify the number of elements in the shape
    using size_type = std::size_t;

    /// No-op for ShapeBase because ShapeBase has no state
    ShapeBase() noexcept = default;

    /// Defaulted polymorphic dtor
    virtual ~ShapeBase() noexcept = default;

    /** @brief Deep polymorphic copy of *this.
     *
     *  @return A pointer to a deep copy of *this.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    base_pointer clone() const { return clone_(); }

    /** @brief The total rank of of the tensor described by *this.
     *
     *  In the simplest terms, the total rank of a tensor is the number of
     *  offsets needed to uniquely distinguish among scalar elements. For
     *  example, a scalar is rank 0 (there is only a single element in the
     *  tensor, so there is no offset needed). A column/row vector is rank 1
     *  because an offset for the row/column is needed. A matrix is rank 2
     *  because offsets for both the row and column are needed, etc.
     *
     *  @return An object containing the rank of the tensor
     *          associated with *this.
     *
     *  @throw None No throw guarantee.
     */
    rank_type rank() const noexcept { return get_rank_(); }

    /** @brief The total number of elements in the tensor described by *this.
     *
     *  Ultimately each tensor is simply a collection of scalar values arranged
     *  into an array. This method is used to determine how many total scalars
     *  are in this array. The total includes both implicit (for example zeros
     *  in sparse data structures) and explicit elements.
     *
     *  @return An object containing the number of elements in *this.
     *
     *  @throw None No throw guarantee.
     */
    size_type size() const noexcept { return get_size_(); }

    /** @brief Polymorphic value comparison.
     *
     *  This method is used to compare two ShapeBase objects polymorphically.
     *  The instances will be cast to their most derived type. If the most
     *  derived types are the same then the objects will be value compared as
     *  derived objects.
     *
     *  @param[in] rhs The object to compare against.
     *
     *  @return True if *this is polymorphically value equal to @p rhs and false
     *          otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool are_equal(const ShapeBase& rhs) const noexcept {
        return are_equal_(rhs) && rhs.are_equal_(*this);
    }

protected:
    /** @brief Used to implement clone()
     *
     *  Derived classes should override this method to implement clone. In
     *  general, if the derived class's copy ctor is a deep copy, then one
     *  simply needs to do:
     *
     *  @code
     *  // Replace DerivedType with the actual type of the derived class
     *  return std::make_unique<DerivedType>(*this);
     *  @endcode
     *
     *  to implement clone_.
     *
     *  @return A deep copy of *this, done polymorphically.
     *
     *  @throw std::bad_alloc if the copy fails. Strong throw guarantee.
     */
    virtual base_pointer clone_() const = 0;

    /** @brief Used to implement rank().
     *
     *  The derived class is responsible for implementing this method so that
     *  it returns a `rank_type` object defining the rank of the derived class.
     *
     *  @return The rank of the derived class.
     *
     *  @throw None Derived classes are responsible for implementing this method
     *              subject to a no-throw guarantee.
     */
    virtual rank_type get_rank_() const noexcept = 0;

    /** @brief Used to implement size().
     *
     *  The derived class is responsible for implementing this method so that
     *  it returns a `size_type` object defining the total number of elements
     *  in the derived class.
     *
     *  @return The total number of elements in the derived class.
     *
     *  @throw None Derived classes are responsible for implementing this method
     *              subject to a no-throw guarantee.
     */
    virtual size_type get_size_() const noexcept = 0;

    /** @brief Called by derived class to implement are_equal_
     *
     *  @tparam DerivedType The type of the derived class for which are_equal_
     *                      is being implemented. Derived class must provide
     *                      this value.
     *
     *  This method is a convience method for implementing are_equal_. Derived
     *  classes need only call this method from their overload of are_equal_
     *  to implement are_equal_.
     *
     *  @param[in] rhs The shape to compare to.
     *
     *  @return True if *this and @p rhs are convertible to DerivedType objects
     *          and if, when viewed as DerivedType objects, *this and @p rhs
     *          are value equal. False otherwise.
     *
     *  @throw None No throw guarantee.
     */
    template<typename DerivedType>
    bool are_equal_impl_(const ShapeBase& rhs) const noexcept {
        auto pthis = dynamic_cast<const DerivedType*>(this);
        auto prhs  = dynamic_cast<const DerivedType*>(&rhs);
        if(pthis == nullptr || prhs == nullptr) return false;
        return (*pthis) == (*prhs);
    }

    /** @brief Derived class overrides to implement are_equal.
     *
     *  Derived classes should implement this method by calling are_equal_impl_.
     *  This assumes that the derived class has implemented a non-polymorphic
     *  value equality check via operator==.
     *
     *  @param[in] rhs The shape to compare to.
     *
     *  @return True if *this is value equal to @p rhs (when compared as objects
     *          of *this most derived type) and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    virtual bool are_equal_(const ShapeBase& rhs) const noexcept = 0;
};

} // namespace tensorwrapper::shape
