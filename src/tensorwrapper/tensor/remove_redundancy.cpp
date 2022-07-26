#include "../ta_helpers/remove_redundancy.hpp"
#include "conversion/conversion.hpp"
#include "detail_/ta_to_tw.hpp"
#include <tensorwrapper/tensor/remove_redundancy.hpp>

namespace tensorwrapper::tensor {

ScalarTensorWrapper remove_redundancy(const ScalarTensorWrapper& C,
                                      const ScalarTensorWrapper& S,
                                      double thresh) {
    to_ta_distarrayd_t converter;
    const auto& C_ta = converter.convert(C.buffer());
    const auto& S_ta = converter.convert(S.buffer());
    auto new_C       = ta_helpers::remove_redundancy(C_ta, S_ta, thresh);
    return detail_::ta_to_tw(std::move(new_C));
}

} // namespace tensorwrapper::tensor
