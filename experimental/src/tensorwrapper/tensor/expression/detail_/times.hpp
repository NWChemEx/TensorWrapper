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

namespace tensorwrapper::tensor::expression::detail_ {

/** @brief Implements multiplication of two expressions
 *
 *  This class holds two expressions, referred to as `a` and `b`, and computes
 *  `a * b` when evaluated.
 *
 *  @tparam FieldType A strong type representing the mathematical field from
 *                    which the tensors' elements are drawn.
 */
template<typename FieldType>
class Times : public Binary<FieldType, Times<FieldType>> {
private:
    /// Type of this class
    using my_type = Times<FieldType>;

    /// Type of the base class
    using base_type = Binary<FieldType, my_type>;

public:
    /// Type of the labels. Ultimately set by FieldTraits<FieldType>::label_type
    using typename base_type::label_type;

    /// Type of labeled views compatible with this expression. Ultimately
    /// set by FieldTraits<FieldType>::const_label_reference
    using typename base_type::const_label_reference;

    /// Type of a read-only allocator. Ultimately set by
    /// FieldTraits<FieldType>::const_allocator_reference
    using typename base_type::const_allocator_reference;

    /// Type of a read-only shape. Ultimately set by
    /// FieldTratis<FieldType>::const_shape_reference
    using typename base_type::const_shape_reference;

    /// Type of the tensor which results from evaluating this expression.
    /// Ultimately set by the value of FieldTraits<FieldType>::tensor_type
    using typename base_type::tensor_type;

    /// Reuses the base class's ctors
    using base_type::base_type;

protected:
    /// Implements labels() by just returning @p lhs_labels
    label_type labels_(const_label_reference lhs_labels) const override {
        return lhs_labels;
    }

    /** @brief Implements tensor by calling Buffer::times
     *
     *  @param[in] lhs_labels The labels for the output tensor.
     *  @param[in] shape The shape of the output tensor.
     *  @param[in] alloc The allocator for the output tensor
     */
    tensor_type tensor_(const_label_reference lhs_labels,
                        const_shape_reference shape,
                        const_allocator_reference alloc) const override;
};

template<typename FieldType>
typename Times<FieldType>::tensor_type Times<FieldType>::tensor_(
  const_label_reference lhs_labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    const auto& exp_a = this->template arg<0>();
    const auto& exp_b = this->template arg<1>();

    const auto a_labels  = exp_a.labels(lhs_labels);
    const auto b_labels  = exp_b.labels(lhs_labels);
    const auto& c_labels = lhs_labels;

    auto a = exp_a.tensor(a_labels, shape, alloc);
    auto b = exp_b.tensor(b_labels, shape, alloc);
    tensor_type c(shape.clone(), alloc.clone());
    auto& c_buffer       = c.buffer();
    const auto& a_buffer = a.buffer();
    const auto& b_buffer = b.buffer();

    a_buffer.times(a_labels, c_labels, c_buffer, b_labels, b_buffer);

    return c;
}

} // namespace tensorwrapper::tensor::expression::detail_
