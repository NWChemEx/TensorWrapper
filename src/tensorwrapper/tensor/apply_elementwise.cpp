#include "../ta_helpers/ta_helpers.hpp"
#include "detail_/ta_to_tw.hpp"
#include "tensorwrapper/tensor/apply_elementwise.hpp"

namespace tensorwrapper::tensor {

using tensor = ScalarTensorWrapper;

tensor apply_elementwise(const tensor& input,
                         const std::function<double(double)>& fxn) {
    const auto& t = input.get<TA::TSpArrayD>();
    return detail_::ta_to_tw(
      tensorwrapper::ta_helpers::apply_elementwise(t, fxn));
}

void apply_elementwise_inplace(tensor& input,
                               const std::function<void(double&)>& fxn) {
    auto& t = input.get<TA::TSpArrayD>();
    tensorwrapper::ta_helpers::apply_elementwise_inplace(t, fxn);
}

} // namespace tensorwrapper::tensor
