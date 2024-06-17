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
#include "nnary.hpp"
#include <tensorwrapper/tensor/expression/labeled_view.hpp>

namespace tensorwrapper::tensor::expression::detail_ {

/** @brief Wraps an end-point for the expression layer
 *
 *  The inputs to the expression layer are labeled tensors. The Labeled class
 *  allows us to wrap these labeled tensors in Expression objects. This class
 *  also handles transposing outside of other expressions.
 *
 *  @tparam FieldType A strong type signaling whether the tensor is filled with
 *                    scalars or tensors. Expected to be either field::Scalar or
 *                    field::Tensor.
 *
 */
template<typename FieldType>
class Labeled : public LabeledBase<FieldType, Labeled<FieldType>> {
private:
    /// Type of *this
    using my_type = Labeled<FieldType>;

    /// Type of the base class
    using base_type = LabeledBase<FieldType, my_type>;

public:
    /// Type of a read-only reference to an Allocator. Ultimately resolves to
    /// FieldTraits<FieldType>::const_allocator_reference
    using typename base_type::const_allocator_reference;

    /// Type of a read-only reference to the labels. Ultimately resolves to
    /// FieldTraits<FieldType>::const_label_reference
    using typename base_type::const_label_reference;

    /// Type of a read-only reference to the Shape. Ultimately resolves to
    /// FieldTraits<FieldType>::const_shape_reference
    using typename base_type::const_shape_reference;

    /// Type of the labels. Resolves to FieldTraits<FieldType>::label_type
    using typename base_type::label_type;

    /// Type of the tensor returned by tensor(). Ultimately resolves to
    /// FieldTraits<FieldType>::tensor_type
    using typename base_type::tensor_type;

    /// Re-uses the base class's ctors
    using base_type::base_type;

protected:
    /// Implements labels() by returning the labels on the wrapped tensor
    label_type labels_(const_label_reference) const override {
        return this->template arg<0>().labels();
    }

    /** @brief Returns the wrapped tensor (possibly transposing)
     *
     *  This function returns the wrapped tensor. If @p lhs_labels are the same
     *  as the labels on the wrapped tensor, then this operation is a no-op. If
     *  the labels are a permutation then the result will be transposed.
     *
     *  @param[in] lhs_labels The labels for the output tensor
     *  @param[in] shape The shape of the output tensor
     *  @param[in] alloc The allocator for the output tensor
     *
     *  @return The wrapped tensor, permuted if @p lhs_labels are a permutation
     *          of the wrapped tensor's labels.
     */
    tensor_type tensor_(const_label_reference labels,
                        const_shape_reference shape,
                        const_allocator_reference alloc) const override;
};

template<typename FieldType>
typename Labeled<FieldType>::tensor_type Labeled<FieldType>::tensor_(
  const_label_reference labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    // Input is b, we're doing b = a
    const auto& a_exp = this->template arg<0>();

    const auto& a_labels = a_exp.labels();
    const auto& a_tensor = a_exp.tensor();
    const auto& a_buffer = a_tensor.buffer();

    tensor_type b(shape.clone(), alloc.clone());
    const auto& b_labels = labels;
    auto& b_buffer       = b.buffer();

    if(b_labels != a_labels) {
        a_buffer.permute(a_labels, b_labels, b_buffer);
    } else {
        b_buffer = a_buffer;
    }

    return b;
}

} // namespace tensorwrapper::tensor::expression::detail_
