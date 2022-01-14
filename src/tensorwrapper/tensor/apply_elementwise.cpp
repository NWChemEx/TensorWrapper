#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/apply_elementwise.hpp"

namespace tensorwrapper::tensor {

using tensor = ScalarTensorWrapper;

tensor apply_elementwise(const tensor& input,
                         const std::function<double(double)>& fxn) {
    const auto& t = input.get<TA::TSpArrayD>();
    return tensor(tensorwrapper::ta_helpers::apply_elementwise(t, fxn));
}

void apply_elementwise_inplace(tensor& input,
                               const std::function<void(double&)>& fxn) {
    auto& t = input.get<TA::TSpArrayD>();
    tensorwrapper::ta_helpers::apply_elementwise_inplace(t, fxn);
}

} // namespace tensorwrapper::tensor
