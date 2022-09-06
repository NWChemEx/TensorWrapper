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

/** @brief Implements scaling of a tensor.
 *
 *  This class implements an expression of the form c = a * b where a and c are
 *  tensors and b is a scalar.
 *
 *  @tparam FieldType A strong type denoting the type of the elements in the
 *                    tensor. Assumed to be either field::Scalar or
 *                    field::Tensor.
 */
template<typename FieldType>
class Scale : public ScaleBase<FieldType, Scale<FieldType>, double> {
private:
    /// Type of *this
    using my_type = Scale<FieldType>;

    /// Type of the base class
    using base_type = ScaleBase<FieldType, my_type, double>;

public:
    /// Type of a read-only reference to an Allocator. Ultimately set by the
    /// value of FieldTraits<FieldType>::const_allocator_reference
    using typename base_type::const_allocator_reference;

    /// Type of a read-only reference to the labels. Ultimately set by the value
    /// of FieldTraits<FieldType>::const_label_reference
    using typename base_type::const_label_reference;

    /// Type of a read-only reference to the Shape. Ultimately set by the value
    /// of FieldTraits<FieldType>::const_shape_reference
    using typename base_type::const_shape_reference;

    /// Type of the labels. Ultimately set by FieldTraits<FieldType>::label_type
    using typename base_type::label_type;

    /// Type of the result. Ultimately set by
    /// FieldTraits<FieldType>::tensor_type
    using typename base_type::tensor_type;

    /// Pull base ctors into scope
    using base_type::NNary;

protected:
    /// Implements labels() by simply returning @p lhs_labels
    label_type labels_(const_label_reference lhs_labels) const override {
        return lhs_labels;
    }

    /** @brief Evaluates the scaling operation held by *this
     *
     *  @param[in] lhs_labels The labels for the output tensor
     *  @param[in] shape The shape of the output tensor
     *  @param[in] alloc The allocator for the output tensor
     *
     *  @return The result of the scaling operation.
     */
    tensor_type tensor_(const_label_reference lhs_labels,
                        const_shape_reference shape,
                        const_allocator_reference alloc) const override;
};

template<typename FieldType>
typename Scale<FieldType>::tensor_type Scale<FieldType>::tensor_(
  const_label_reference lhs_labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    // We're doing c = a * b (b is the scalar)

    const auto& exp_a = this->template arg<0>();

    const auto a_labels  = exp_a.labels(lhs_labels);
    const auto& c_labels = lhs_labels;

    auto a = exp_a.tensor(a_labels, shape, alloc);
    auto b = this->template arg<1>();
    tensor_type c(shape.clone(), alloc.clone());

    auto& a_buffer = a.buffer();
    auto& c_buffer = c.buffer();

    a_buffer.scale(a_labels, c_labels, c_buffer, b);

    return c;
}

} // namespace tensorwrapper::tensor::expression::detail_
