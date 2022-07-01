#pragma once
#include "nnary.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expression::detail_ {

/** @brief Implements subtraction of two expressions
 *
 *  This class holds two expressions, referred to as `a` and `b`, and computes
 *  `a - b` when evaluated.
 *
 *  @tparam FieldType A strong type representing the mathematical field from
 *                    which the tensors' elements are drawn.
 */
template<typename FieldType>
class Subtract : public Binary<FieldType, Subtract<FieldType>> {
private:
    /// Type of this class
    using my_type = Subtract<FieldType>;

    /// Type of the base class
    using base_type = Binary<FieldType, my_type>;

public:
    using typename base_type::label_type;

    /// Type of labeled views compatible with this expression. Ultimately
    /// set by the FieldTraits<FieldType>::const_label_reference
    using typename base_type::const_label_reference;

    using typename base_type::const_allocator_reference;

    using typename base_type::const_shape_reference;

    /// Type of the tensor which results from evaluating this expression.
    /// Ultimately set by the value of Expression<FieldType>::tensor_type
    using typename base_type::tensor_type;

    /// Reuses the base class's ctors
    using base_type::NNary;

protected:
    label_type labels_(const_label_reference lhs_labels) const override {
        return lhs_labels;
    }

    /** @brief Implements tensor by calling Buffer::subtract
     *
     *  @param[in] lhs A labeled tensor containing the details
     */
    tensor_type tensor_(const_label_reference lhs_labels,
                        const_shape_reference shape,
                        const_allocator_reference alloc) const override;
};

template<typename FieldType>
typename Subtract<FieldType>::tensor_type Subtract<FieldType>::tensor_(
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

    a_buffer.subtract(a_labels, c_labels, c_buffer, b_labels, b_buffer);

    return c;
}

} // namespace tensorwrapper::tensor::expression::detail_
