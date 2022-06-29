#pragma once
#include "nnary.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expression::detail_ {

template<typename FieldType>
class Add : public Binary<FieldType, Add<FieldType>> {
private:
    using my_type   = Add<FieldType>;
    using base_type = Binary<FieldType, my_type>;

public:
    using typename base_type::labeled_tensor;
    using typename base_type::pimpl_pointer;

    using base_type::NNary;

protected:
    labeled_tensor& eval_(labeled_tensor& lhs) const override;
};

template<typename FieldType>
typename Add<FieldType>::labeled_tensor& Add<FieldType>::eval_(
  labeled_tensor& result) const {
    labeled_tensor temp_l(result), temp_r(result);
    temp_l = this->template arg<0>().eval(temp_l);
    temp_r = this->template arg<1>().eval(temp_r);

    const auto& result_labels = result.labels();
    const auto& l_labels      = temp_l.labels();
    const auto& r_labels      = temp_r.labels();

    auto& result_buffer = result.tensor().buffer();
    const auto& lbuffer = temp_l.tensor().buffer();
    const auto& rbuffer = temp_r.tensor().buffer();

    lbuffer.add(l_labels, result_labels, result_buffer, r_labels, rbuffer);

    return result;
}

} // namespace tensorwrapper::tensor::expression::detail_
