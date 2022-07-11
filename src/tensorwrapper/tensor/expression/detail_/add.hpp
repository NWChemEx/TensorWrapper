#pragma once
#include "nnary.hpp"

namespace tensorwrapper::tensor::expression::detail_ {

/** @brief Implements addition of two expressions
 *
 *  This class holds two expressions, referred to as `a` and `b`, and computes
 *  `a + b` when evaluated.
 *
 *  @tparam FieldType A strong type representing the mathematical field from
 *                    which the tensors' elements are drawn.
 */
template<typename FieldType>
class Add : public Binary<FieldType, Add<FieldType>> {
private:
    /// Type of this class
    using my_type = Add<FieldType>;

    /// Type of the base class
    using base_type = Binary<FieldType, my_type>;

public:
    /// Type used for labeling tensor modes. Ultimately set by
    /// FieldTraits<FieldType>::label_type
    using typename base_type::label_type;

    /// Type of labeled views compatible with this expression. Ultimately
    /// set by the FieldTraits<FieldType>::const_label_reference
    using typename base_type::const_label_reference;

    /// Type of a read-only reference to an Allocator. Ultimately set by
    /// FieldTraits<FieldType>::const_allocator_reference
    using typename base_type::const_allocator_reference;

    /// Type of a read-only reference to a Shape. Ultimately set by
    /// FieldTraits<FieldType>::const_shape_reference
    using typename base_type::const_shape_reference;

    /// Type of the tensor which results from evaluating this expression.
    /// Ultimately set by the value of Expression<FieldType>::tensor_type
    using typename base_type::tensor_type;

    /// Reuses the base class's ctors
    using base_type::NNary;

protected:
    /// Implements labels() by returning the input labels
    label_type labels_(const_label_reference lhs_labels) const override {
        return lhs_labels;
    }

    /** @brief Implements tensor by calling Buffer::add
     *
     *  @param[in] lhs_labels The output tensor's labels
     *  @param[in] shape      The output tensor's shape
     *  @param[in] alloc      The output tensor's allocator
     *
     *  @return The result of adding the wrapped expressions together according
     *          to the annotations.
     */
    tensor_type tensor_(const_label_reference lhs_labels,
                        const_shape_reference shape,
                        const_allocator_reference alloc) const override;
};

template<typename FieldType>
typename Add<FieldType>::tensor_type Add<FieldType>::tensor_(
  const_label_reference lhs_labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    // For notational purposes we assume we're doing c = a + b

    // Unpack the expressions for a and b from base class
    const auto& exp_a = this->template arg<0>();
    const auto& exp_b = this->template arg<1>();

    // Get the labels for the three tensors
    const auto a_labels  = exp_a.labels(lhs_labels);
    const auto b_labels  = exp_b.labels(lhs_labels);
    const auto& c_labels = lhs_labels;

    // Evaluate a and  b into tensors, initialize c
    auto a = exp_a.tensor(a_labels, shape, alloc);
    auto b = exp_b.tensor(b_labels, shape, alloc);
    tensor_type c(shape.clone(), alloc.clone());

    // Do the addition
    auto& c_buffer       = c.buffer();
    const auto& a_buffer = a.buffer();
    const auto& b_buffer = b.buffer();
    a_buffer.add(a_labels, c_labels, c_buffer, b_labels, b_buffer);

    return c;
}

} // namespace tensorwrapper::tensor::expression::detail_
