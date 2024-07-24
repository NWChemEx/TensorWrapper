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
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/shape/shape_base.hpp>
#include <tensorwrapper/sparsity/pattern.hpp>
#include <tensorwrapper/symmetry/group.hpp>

namespace tensorwrapper::layout {

/** @brief Common base class for all layouts.
 *
 */
class LayoutBase : public detail_::PolymorphicBase<LayoutBase> {
public:
    /// Type all layouts derive from
    using layout_base = LayoutBase;

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

    /// Type of a pointer to an object of type symmetry_type
    using symmetry_pointer = std::unique_ptr<symmetry_type>;

    /// Object holding sparsity patterns
    using sparsity_type = sparsity::Pattern;

    /// Read-only reference to the sparsity
    using const_sparsity_reference = const sparsity_type&;

    /// Type of a pointer to an object of type sparsity_type
    using sparsity_pointer = std::unique_ptr<sparsity_type>;

    /// Type used for indexing and offsets
    using size_type = std::size_t;

    // -------------------------------------------------------------------------
    // -- Ctors and dtor
    // -------------------------------------------------------------------------

    /** @brief Initialize by copy ctor
     *
     *  This ctor is used when the user does not want to relinquish ownership of
     *  the objects used to initialize *this. The inputs will be copied.
     *
     *  @param[in] shape The actual shape the tensor backend has.
     *  @param[in] symmetry The actual symmetry the backend has.
     *  @param[in] sparsity The actual sparsity the backend has.
     *
     *  @throw std::bad_alloc if there is a problem allocating the new state.
     *                        Strong throw guarantee.
     */
    LayoutBase(const_shape_reference shape, const_symmetry_reference symmetry,
               const_sparsity_reference sparsity) :
      LayoutBase(shape.clone(), std::make_unique<symmetry_type>(symmetry),
                 std::make_unique<sparsity_type>(sparsity)) {}

    /** @brief Initialize by move ctor
     *
     *  This ctor is used when the user wants *this to take ownership of the
     *  objects being used to initialize * this.
     *
     *  @throw std::runtime_error if @p shape, @p symmetry, or @p sparsity is
     *                            a nullptr. Strong throw guarantee.
     */
    LayoutBase(shape_pointer shape, symmetry_pointer symmetry,
               sparsity_pointer sparsity) :
      m_shape_(std::move(shape)),
      m_symmetry_(std::move(symmetry)),
      m_sparsity_(std::move(sparsity)) {
        if(m_shape_ == nullptr) throw std::runtime_error("Shape can't be null");
        if(m_symmetry_ == nullptr)
            throw std::runtime_error("Symmetry can't be null");
        if(m_sparsity_ == nullptr)
            throw std::runtime_error("Sparsity can't be null");
    }

    /// Defaulted polymorphic dtor
    virtual ~LayoutBase() noexcept = default;

    // -------------------------------------------------------------------------
    // -- State methods
    // -------------------------------------------------------------------------

    /** @brief Provides read-only access to the shape of the layout.
     *
     *  @return A read-only reference to the shape of the layout.
     *
     *  @throw None No throw guarantee
     */
    const_shape_reference shape() const { return *m_shape_; }

    /** @brief Provides read-only access to the symmetry of the layout.
     *
     *  @return A read-only reference to the symmetry of the layout.
     *
     *  @throw None No throw guarantee
     */
    const_symmetry_reference symmetry() const { return *m_symmetry_; }

    /** @brief Provides access to the sparsity of the layout.
     *
     *  @return A read-only reference to the sparsity of the layout.
     *
     *  @throw None No throw guarantee.
     */
    const_sparsity_reference sparsity() const { return *m_sparsity_; }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Is *this value equal to @p rhs?
     *
     *  Two LayoutBase objects are value equal if they both don't have shapes or
     * if they have the same shapes, symmetry, and sparsity.
     *
     *  @param[in] rhs The object to compare *this to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const layout_base& rhs) const noexcept {
        if(!m_shape_->are_equal(*rhs.m_shape_)) return false;
        return std::tie(*m_symmetry_, *m_sparsity_) ==
               std::tie(*rhs.m_symmetry_, *rhs.m_sparsity_);
    }

    /** @brief Is *this different from @p rhs?
     *
     *  Two LayoutBase objects are different if they are not value equal. This
     * method simply negates operator==. See the description of operator== for
     * the definition of value equal.
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
    LayoutBase(const LayoutBase& other) :
      m_shape_(other.m_shape_->clone()),
      m_symmetry_(std::make_unique<symmetry_type>(*other.m_symmetry_)),
      m_sparsity_(std::make_unique<sparsity_type>(*other.m_sparsity_)) {}

    LayoutBase& operator=(const LayoutBase&) = delete;
    LayoutBase& operator=(LayoutBase&&) = delete;

private:
    /// The actual shape of the tensor
    shape_pointer m_shape_;

    /// The actual symmetry of the tensor
    symmetry_pointer m_symmetry_;

    /// The actual sparsity of the tensor
    sparsity_pointer m_sparsity_;
};

} // namespace tensorwrapper::layout
