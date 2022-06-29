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
    using labeled_tensor = typename base_type::labeled_tensor;

    using base_type::NNary;

protected:
    labeled_tensor& eval_(labeled_tensor& result) const override;
};

template<typename FieldType>
typename Scale<FieldType>::labeled_tensor& Scale<FieldType>::eval_(
  labeled_tensor& result) const {
    labeled_tensor temp(result);
    temp = this->template arg<0>().eval(temp);

    const auto& rlabels = temp.labels();
    const auto& llabels = result.labels();

    auto& rbuffer = temp.tensor().buffer();
    auto& lbuffer = result.tensor().buffer();

    rbuffer.scale(rlabels, llabels, lbuffer, this->template arg<1>());

    return result;
}

} // namespace tensorwrapper::tensor::expression::detail_
