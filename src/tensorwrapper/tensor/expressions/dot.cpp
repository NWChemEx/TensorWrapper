#include <tensorwrapper/tensor/expressions/dot.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expressions {

double dot(const LabeledTensor<field::Scalar>& lhs,
           const LabeledTensor<field::Scalar>& rhs) {
    const auto& llabels = lhs.labels();
    const auto& rlabels = rhs.labels();
    const auto& lbuffer = lhs.tensor().buffer();
    const auto& rbuffer = rhs.tensor().buffer();

    return lbuffer.dot(llabels, rlabels, rbuffer);
}

} // namespace tensorwrapper::tensor::expressions
