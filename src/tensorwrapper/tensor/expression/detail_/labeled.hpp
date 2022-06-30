#pragma once
#include "nnary.hpp"
#include <tensorwrapper/tensor/expression/labeled_view.hpp>

namespace tensorwrapper::tensor::expression::detail_ {

template<typename FieldType>
class Labeled : public LabeledBase<FieldType, Labeled<FieldType>> {
private:
    using my_type   = Labeled<FieldType>;
    using base_type = LabeledBase<FieldType, my_type>;

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

private:
    const auto& tensor() const { return this->template arg<0>(); }
};

template<typename FieldType>
typename Labeled<FieldType>::tensor_type Labeled<FieldType>::tensor_(
  const_label_reference labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    // Input is b, we're doing b = a

    const auto& a_labels = tensor().labels();
    const auto& a_tensor = tensor().tensor();
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
