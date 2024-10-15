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
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/shape/shape_traits.hpp>
#include <tensorwrapper/shape/smooth_view.hpp>

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
class ShapeBase : public tensorwrapper::detail_::PolymorphicBase<ShapeBase> {
private:
    /// Type implementing the traits of this
    using traits_type = ShapeTraits<ShapeBase>;

public:
    /// Type all shapes inherit from
    using shape_base = typename traits_type::shape_base;

    /// Type of a pointer to the base of a shape object
    using base_pointer = typename traits_type::base_pointer;

    /// Type used to hold the rank of a tensor
    using rank_type = typename traits_type::rank_type;

    /// Type used to specify the number of elements in the shape
    using size_type = typename traits_type::size_type;

    /// Type of an object acting like a mutable reference to a Smooth shape
    using smooth_reference = SmoothView<Smooth>;

    /// Type of an object acting like a read-only reference to a Smooth shape
    using const_smooth_reference = SmoothView<const Smooth>;

    /// No-op for ShapeBase because ShapeBase has no state
    ShapeBase() noexcept = default;

    /// Defaulted polymorphic dtor
    virtual ~ShapeBase() noexcept = default;

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

    /** @brief Returns a view of *this as a Smooth object.
     *
     *  It is possible to view any shape as a smooth shape. For more exotic
     *  shapes this may require flattening nestings and padding dimensions.
     *  This method ultimately dispatches to the as_smooth_ overload of the
     *  derived class to control how to smooth the shape out.
     *
     *  @return A view of *this consistent with thinking of *this as a Smooth
     *          object.
     *
     *  @throw std::bad_alloc if there is a problem allocating the view. Strong
     *                        throw guarantee.
     */
    smooth_reference as_smooth() { return as_smooth_(); }

    /** @brief Returns a read-only view of *this as a Smooth object.
     *
     *  This method works the same as the non-const version except that the
     *  resulting view is read-only.
     *
     *  @return A read-only view of *this consistent with thinking of *this as
     *          a Smooth object.
     *
     *  @throw std::bad_alloc if there is a problem allocating the view. Strong
     *                        throw guarantee.
     */
    const_smooth_reference as_smooth() const { return as_smooth_(); }

protected:
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

    /// Derived class should override to be consistent with as_smooth()
    virtual smooth_reference as_smooth_() = 0;

    /// Derived class should override to be consistent with as_smooth() const
    virtual const_smooth_reference as_smooth_() const = 0;
};

} // namespace tensorwrapper::shape
