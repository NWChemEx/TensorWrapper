#include "../ta_helpers/remove_redundancy.hpp"
#include "detail_/ta_to_tw.hpp"
#include "tensorwrapper/tensor/remove_redundancy.hpp"

namespace tensorwrapper::tensor {

ScalarTensorWrapper remove_redundancy(const ScalarTensorWrapper& C,
                                      const ScalarTensorWrapper& S,
                                      double thresh) {
    const auto& C_ta = C.get<TA::TSpArrayD>();
    const auto& S_ta = S.get<TA::TSpArrayD>();
    auto new_C       = ta_helpers::remove_redundancy(C_ta, S_ta, thresh);
    return detail_::ta_to_tw(std::move(new_C));
}

} // namespace tensorwrapper::tensor
