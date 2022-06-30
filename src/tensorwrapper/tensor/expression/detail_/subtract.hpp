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
    /** @brief Implements tensor by calling Buffer::subtract
     *
     *  @param[in] lhs A labeled tensor containing the details
     */
    tensor_type tensor_(const_label_reference labels,
                        const_shape_reference shape,
                        const_allocator_reference alloc) const override;
};

template<typename FieldType>
typename Subtract<FieldType>::tensor_type Subtract<FieldType>::tensor_(
  const_label_reference labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    // TODO This is going to transpose a and b if they don't match c, we'd
    //      rather it happen as part of the call to buffer.

    const auto& a_labels = labels;
    const auto& b_labels = labels;
    const auto& c_labels = labels;

    auto a = this->template arg<0>().tensor(c_labels, shape, alloc);
    auto b = this->template arg<1>().tensor(c_labels, shape, alloc);

    tensor_type c(shape.clone(), alloc.clone());
    auto& c_buffer       = c.buffer();
    const auto& a_buffer = a.buffer();
    const auto& b_buffer = b.buffer();

    a_buffer.subtract(a_labels, c_labels, c_buffer, b_labels, b_buffer);

    return c;
}

} // namespace tensorwrapper::tensor::expression::detail_
