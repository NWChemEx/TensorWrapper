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
#include <tensorwrapper/shape/shape_base.hpp>
#include <tensorwrapper/sparsity/pattern.hpp>
#include <tensorwrapper/symmetry/group.hpp>

namespace tensorwrapper::layout {

/** @brief Describes how the tensor is actually laid out.
 *
 */
class Tiled {
public:
    /// Type all layouts derive from
    using layout_base = Tiled;

    /// Type of a mutable reference to the base of a layout
    using layout_reference = layout_base&;

    /// Type of a read-only reference to the base of a layout
    using const_layout_reference = const layout_base&;

    /// Type of a pointer to the base of a layout
    using layout_pointer = std::unique_ptr<layout_base>;

    /// Common base type of all shape objects
    using shape_base = shape::ShapeBase;

    /// Read-only reference to a shape's base object.
    using const_shape_reference = const shape_base&;

    /// Pointer to the base type of a shape object
    using shape_pointer = typename shape_base::base_pointer;

    /// Object holding symmetry operations
    using symmetry_type = symmetry::Group;

    /// Read-only reference to the symmetry
    using const_symmetry_reference = const symmetry_type&;

    /// Object holding sparsity patterns
    using sparsity_type = sparsity::Pattern;

    /// Read-only reference to the sparsity
    using const_sparsity_reference = const sparsity_type&;

    /// Type used for indexing and offsets
    using size_type = std::size_t;

    // -------------------------------------------------------------------------
    // -- Ctors, assignment, and dtor
    // -------------------------------------------------------------------------

    /** @brief Creates the layout of a defaulted tensor.
     *
     *  Defaulted layouts have no shape, defaulted symmetry, and defaulted
     *  sparsity. Such a layout is consistent with a tensor with no state.
     *
     *  @throw None No throw guarantee.
     */
    Tiled() = default;

    /** @brief Value ctor
     *
     *  @param[in] shape The actual shape the tensor backend has.
     *  @param[in] symmetry The actual symmetry the backend has.
     *  @param[in] sparsity The actual sparsity the backend has.
     *
     *  @throw std::bad_alloc if there is a problem allocating the new state.
     *                        Strong throw guarantee.
     */
    Tiled(const_shape_reference shape, symmetry_type symmetry,
          sparsity_type sparsity) :
      Tiled(shape.clone(), std::move(symmetry), std::move(sparsity)) {}

    /// Defaulted polymorphic dtor
    virtual ~Tiled() noexcept = default;

    /** @brief Make a polymorphic deep copy of *this.
     *
     *  @return A pointer to the deep copy of *this.
     *
     *  @throw std::bad_alloc if allocating the copy fails.
     */
    layout_pointer clone() const { return clone_(); }

    // -------------------------------------------------------------------------
    // -- State methods
    // -------------------------------------------------------------------------

    /** @brief How many tiles does *this have?
     *
     *  Layouts are in general tiled in some manner
     *
     *  @return The number of tiles in *this.
     *
     *  @throw None No throw guarantee.
     */
    size_type tile_size() const noexcept {
        return has_shape() ? tile_size_() : 0;
    }

    /** @brief Does *this have a shape set?
     *
     *  @return True if *this has a shape and false otherwise.
     *
     *  @throw None No throw guarantee
     */
    bool has_shape() const noexcept { return m_shape_ != nullptr; }

    /** @brief Provides read-only access to the symmetry of the layout.
     *
     *  @return A red-only reference to the symmetry of the layout.
     *
     *  @throw None No throw guarantee.
     */
    const_symmetry_reference symmetry() const noexcept { return m_symmetry_; }

    /** @brief Provides access to the sparsity of the layout.
     *
     *  @return A read-only reference to the sparsity of the layout.
     *
     *  @throw None No throw guarantee.
     */
    const_sparsity_reference sparsity() const noexcept { return m_sparsity_; }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Is *this polymorphically value equal to @p rhs?
     *
     *  This method is used to compare *this to @p rhs polymorphically. More
     *  specifically both *this and @p rhs will be downcasted to their most
     *  derived class. If the most derived class of both *this and @p rhs is
     *  the same, then *this will be compared to @p rhs via the most derived
     *  class's value comparison operator. If *this and @p rhs do not have the
     *  same most derived class this method returns false.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this is polymorphically value equal to @p rhs and
     *          false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool are_equal(const layout_base& rhs) const noexcept {
        return are_equal_(rhs) && rhs.are_equal_(*this);
    }

    /** @brief Is *this value equal to @p rhs?
     *
     *  Two Tiled objects are value equal if they both don't have shapes or if
     *  they have the same shapes, symmetry, and sparsity.
     *
     *  @param[in] rhs The object to compare *this to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const layout_base& rhs) const noexcept {
        if(has_shape() != rhs.has_shape()) return false;
        if(!has_shape()) return true;
        if(!m_shape_->are_equal(*rhs.m_shape_)) return false;
        return std::tie(m_symmetry_, m_sparsity_) ==
               std::tie(rhs.m_symmetry_, rhs.m_sparsity_);
    }

    /** @brief Is *this different from @p rhs?
     *
     *  Two Tiled objects are different if they are not value equal. This method
     *  simply negates operator==. See the description of operator== for the
     *  definition of value equal.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const layout_base& rhs) const noexcept {
        return !((*this) == rhs);
    }

protected:
    /** @brief Makes a deep copy of *this.
     *
     *  This method is protected to avoid slicing.
     *
     *  @param[in] other The object to copy.
     *
     *  @throw std::bad_alloc if there is a problem copying @p other. Strong
     *                        throw guarantee.
     */
    Tiled(const Tiled& other) :
      m_shape_(other.has_shape() ? other.m_shape_->clone() : nullptr),
      m_symmetry_(other.m_symmetry_),
      m_sparsity_(other.m_sparsity_) {}

    /** @brief Implements clone
     *
     *  Derived classes should override this method so that it makes a deep
     *  copy of *this via the derived class's copy ctor.
     *
     *  @return A deep copy of *this.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    virtual layout_pointer clone_() const = 0;

    /** @brief Implements tile_size.
     *
     *  For now this is an abstract method. When tiling is actually supported
     *  this method will be implemented in this class. This method is only
     *  called if m_shape_ is non-null (if it's null then we have no tiles).
     *
     *  @return The number of tiles in *this.
     *
     *  @throw None No throw guarantee.
     */
    virtual size_type tile_size_() const noexcept = 0;

    /** @brief Helps derived classes implement are_equal.
     *
     *  @tparam DerivedClass Type of the class implementing are_equal. Must be
     *                       provided by the caller.
     *
     *  This method wraps the process of downcasting *this and @p rhs to objects
     *  of type @p DerivedClass and then comparing them. This method will also
     *  handle the logic for when the downcasts fail.
     *
     *  @param[in] rhs The object to compare against.
     *
     *  @return True if *this and @p rhs compare value equal when compared as
     *          objects of type @p DerivedClass and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    template<typename DerivedType>
    bool are_equal_impl_(const layout_base& rhs) const noexcept {
        auto plhs = dynamic_cast<const DerivedType*>(this);
        auto prhs = dynamic_cast<const DerivedType*>(&rhs);
        if(plhs == nullptr || prhs == nullptr) return false;
        return (*plhs) == (*prhs);
    }

    /** @brief Implements are_equal
     *
     *  Derived classes should override this method so that it calls
     *  are_equal_impl_.
     *
     *  @param[in] rhs The object to compare to.
     *
     *  @return True if *this compared via its most derived class is value equal
     *          to @p rhs as its most derived class and false otherwise.
     *
     *  @throw None No throw guarantee
     */
    virtual bool are_equal_(const layout_base& rhs) const noexcept = 0;

private:
    /// Ctor all other value ctors dispatch to
    Tiled(shape_pointer shape, symmetry_type symmetry, sparsity_type sparsity) :
      m_shape_(std::move(shape)),
      m_symmetry_(std::move(symmetry)),
      m_sparsity_(std::move(sparsity)) {}

    /// The actual shape of the tensor
    shape_pointer m_shape_;

    /// The actual symmetry of the tensor
    symmetry_type m_symmetry_;

    /// The actual sparsity of the tensor
    sparsity_type m_sparsity_;
};

} // namespace tensorwrapper::layout
