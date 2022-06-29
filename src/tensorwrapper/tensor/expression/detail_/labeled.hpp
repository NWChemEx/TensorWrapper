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
    using typename base_type::labeled_tensor;

    using base_type::NNary;

protected:
    labeled_tensor& eval_(labeled_tensor& lhs) const override;

private:
    const auto& tensor() const { return this->template arg<0>(); }
};

template<typename FieldType>
typename Labeled<FieldType>::labeled_tensor& Labeled<FieldType>::eval_(
  labeled_tensor& result) const {
    const auto& rhs_labels = tensor().labels();
    const auto& lhs_labels = result.labels();
    const auto& rhs_buffer = tensor().tensor().buffer();
    auto& lhs_buffer       = result.tensor().buffer();

    if(rhs_labels != lhs_labels) {
        rhs_buffer.permute(rhs_labels, lhs_labels, lhs_buffer);
    } else {
        lhs_buffer = rhs_buffer;
    }

    return result;
}

} // namespace tensorwrapper::tensor::expression::detail_
