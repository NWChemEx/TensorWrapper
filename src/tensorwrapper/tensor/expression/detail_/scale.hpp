#pragma once
#include "nnary.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expression::detail_ {

template<typename FieldType>
class Scale : public ScaleBase<FieldType, Scale<FieldType>, double> {
private:
    using my_type   = Scale<FieldType>;
    using base_type = ScaleBase<FieldType, my_type, double>;

public:
    using typename base_type::const_allocator_reference;
    using typename base_type::const_label_reference;
    using typename base_type::const_shape_reference;
    using typename base_type::label_type;
    using typename base_type::tensor_type;

    using base_type::NNary;

protected:
    label_type labels_(const_label_reference lhs_labels) const override {
        return lhs_labels;
    }

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
