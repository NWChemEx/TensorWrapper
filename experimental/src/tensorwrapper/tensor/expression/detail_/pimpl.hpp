/*
 * Copyright 2022 NWChemEx-Project
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
#include <stdexcept>
#include <tensorwrapper/tensor/expression/expression_class.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expression::detail_ {

/** @brief Implements Expression
 *
 *  The ExpressionPIMPL class implements the Expression class using
 *  type-erasure. More specifically, derived classes such as Add, Scale, and
 *  Subtract override the labels_ and tensor_ methods for their particular
 *  scenario. For example Add overrides labels_ to simply return the provided
 *  labels (permutations are handled internally so the labels can be returned as
 *  requested) and tensor_ is overridden to return the result of adding together
 *  the two sub expressions.
 *
 *  @tparam FieldType A strong type denoting whether the tensor is filled with
 *          scalars or other tensors. Assumed to be either field::Scalar or
 *          field::Tensor.
 */
template<typename FieldType>
class ExpressionPIMPL {
private:
    /// Type of the Expression instance we are implementing, pt stands for
    /// "parent type" and is chosen to avoid the public typedefs occupying
    /// multiple lines
    using pt = Expression<FieldType>;

public:
    /// Type used to label a tensor's modes. Ultimately set by
    /// FieldTraits<FieldType>::label_type
    using label_type = typename pt::label_type;

    /// Type of a read-only reference to the labels. Ultimately set by
    /// FieldTraits<FieldType>::const_label_reference
    using const_label_reference = typename pt::const_label_reference;

    /// Type of a read-only allocator. Ultimately set by
    /// FieldTraits<FieldType>::const_allocator_reference
    using const_allocator_reference = typename pt::const_allocator_reference;

    /// Type of a read-only reference to a Shape. Ultimately set by
    /// FieldTraits<FieldType>::const_shape_reference
    using const_shape_reference = typename pt::const_shape_reference;

    /// Type of a tensor. Set by FieldTraits<FieldType>::tensor_type
    using tensor_type = typename pt::tensor_type;

    /// Type of a pointer to an ExpressionPIMPL instance. Ultimately set by
    /// Expression<FieldType>::pimpl_pointer.
    using pimpl_pointer = typename pt::pimpl_pointer;

    /** @brief Creates an ExpressionPIMPL instance.
     *
     *  ExpressionPIMPL instances have no state (all state is contained in the
     *  derived classes). Hence this ctor is a no-op.
     *
     *  @throw None No throw guarantee.
     */
    ExpressionPIMPL() noexcept = default;

    /// Default, no-throw, polymorphic dtor
    virtual ~ExpressionPIMPL() noexcept = default;

    /** @brief Polymorphic deep copy.
     *
     *  The clone method creates a deep copy of *this's most derived class (the
     *  deep copy includes all state in between ExpressionPIMPL and the derived
     *  class as well). The result is returned as a pointer to the base class.
     *
     *  @return A polymorphic deep copy of *this.
     *
     *  @throw std::bad_alloc if there is a problem allocating memory. Strong
     *                        throw guarantee.
     */
    pimpl_pointer clone() const { return clone_(); }

    /** @brief Determines the labels for assigning *this to a tensor.
     *
     *  This method implements Expression::labels, see the documentation for
     *  Expression::labels for more information.
     */
    label_type labels(const_label_reference lhs_labels) const {
        return labels_(lhs_labels);
    }

    /** @brief Evaluates *this into a tensor.
     *
     *  This method implements Expression::tensor, see Expression::tensor's
     *  documentation for more details.
     */
    tensor_type tensor(const_label_reference labels,
                       const_shape_reference shape,
                       const_allocator_reference alloc) const {
        return tensor_(labels, shape, alloc);
    }

    /** @brief Checks for polymorphic value equality
     *
     *  This method ultimately implements Expression::operator== and
     *  Expression::operator!=. Unlike the aforementioned methods, are_equal
     *  must contend with the fact that it ExpressionPIMPL is polymorphic.
     *  To that extent, are_equal calls the implementing hook are_equal_
     *  symmetrically (this ensures both *this and rhs have the same most
     *  derived class).
     *
     *  Assume D is a class derived from ExpressionPIMPL, D implements are_equal
     *  by downcasting @p rhs to a D instance and then comparing the state of
     *  *this to the state in the part of @p rhs implemented by D. If the
     *  downcast fails, or the state compares different this method returns
     *  false. If the downcast works and the state compares equal, D dispatches
     *  to the base class's implementation of are_equal_ and the process
     *  repeats recursively with the base class taking on the role of D.
     *
     *  @param[in] rhs The ExpressionPIMPL we are comparing to *this.
     *
     *  @return True if *this compares value equal to @p rhs polymorphically and
     *          false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool are_equal(const ExpressionPIMPL& rhs) const noexcept {
        return are_equal_(rhs) && rhs.are_equal_(*this);
    }

protected:
    /// No-op copy ctor, used by derived classes to implement clone
    ExpressionPIMPL(const ExpressionPIMPL& other) = default;

    /// Derived classes should override to implement clone()
    virtual pimpl_pointer clone_() const = 0;

    /// Derived classes should override to implement labels()
    virtual label_type labels_(const_label_reference lhs_labels) const = 0;

    /// Derived classes should override to implement tensor()
    virtual tensor_type tensor_(const_label_reference labels,
                                const_shape_reference shape,
                                const_allocator_reference alloc) const = 0;

    /// Derived classes should override to implement are_equal
    virtual bool are_equal_(const ExpressionPIMPL& rhs) const noexcept = 0;

private:
    // These are deleted to avoid accidentally slicing by copying/moving
    ExpressionPIMPL(ExpressionPIMPL&& other) = delete;
    ExpressionPIMPL& operator=(const ExpressionPIMPL& other) = delete;
    ExpressionPIMPL& operator=(ExpressionPIMPL&& other) = delete;
};

} // namespace tensorwrapper::tensor::expression::detail_
