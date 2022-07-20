#include "../ta_helpers/ta_helpers.hpp"
#include "conversion/conversion.hpp"
#include <tensorwrapper/tensor/allclose.hpp>

namespace tensorwrapper::tensor {

bool allclose(const ScalarTensorWrapper& actual, const ScalarTensorWrapper& ref,
              double rtol, double atol) {
    to_ta_distarrayd_t converter;
    const auto& a = converter.convert(actual.buffer());
    const auto& r = converter.convert(ref.buffer());

    return ta_helpers::allclose(a, r, false, rtol, atol);
}

bool allclose(const TensorOfTensorsWrapper& actual,
              const TensorOfTensorsWrapper& ref, double rtol, double atol) {
    to_ta_totd_t converter;
    const auto& a = converter.convert(actual.buffer());
    const auto& r = converter.convert(ref.buffer());

    auto inner_rank = actual.rank() - a.trange().rank();
    return ta_helpers::allclose_tot(a, r, inner_rank, false, rtol, atol);
}

bool abs_allclose(const ScalarTensorWrapper& actual,
                  const ScalarTensorWrapper& ref, double rtol, double atol) {
    to_ta_distarrayd_t converter;
    const auto& a = converter.convert(actual.buffer());
    const auto& r = converter.convert(ref.buffer());

    return ta_helpers::allclose(a, r, true, rtol, atol);
}

bool abs_allclose(const TensorOfTensorsWrapper& actual,
                  const TensorOfTensorsWrapper& ref, double rtol, double atol) {
    to_ta_totd_t converter;
    const auto& a = converter.convert(actual.buffer());
    const auto& r = converter.convert(ref.buffer());

    auto inner_rank = actual.rank() - a.trange().rank();
    return ta_helpers::allclose_tot(a, r, inner_rank, true, rtol, atol);
}

} // namespace tensorwrapper::tensor
