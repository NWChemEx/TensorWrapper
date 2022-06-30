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
    using typename base_type::tensor_type;

    using base_type::NNary;

protected:
    tensor_type tensor_(const_label_reference labels,
                        const_shape_reference shape,
                        const_allocator_reference alloc) const override;
};

template<typename FieldType>
typename Scale<FieldType>::tensor_type Scale<FieldType>::tensor_(
  const_label_reference labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    // We're doing c = a * b (b is the scalar)

    auto a = this->template arg<0>().tensor(labels, shape, alloc);
    auto b = this->template arg<1>();
    tensor_type c(shape.clone(), alloc.clone());

    const auto& a_labels = labels;
    const auto& c_labels = labels;

    auto& a_buffer = a.buffer();
    auto& c_buffer = c.buffer();

    a_buffer.scale(a_labels, c_labels, c_buffer, b);

    return c;
}

} // namespace tensorwrapper::tensor::expression::detail_
