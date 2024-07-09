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
class ShapeBase : public detail_::PolymorphicBase<ShapeBase> {
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
};

} // namespace tensorwrapper::shape
