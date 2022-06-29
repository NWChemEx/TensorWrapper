#include <tensorwrapper/tensor/expression/dot.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expression {

double dot(const LabeledView<field::Scalar>& lhs,
           const LabeledView<field::Scalar>& rhs) {
    const auto& llabels = lhs.labels();
    const auto& rlabels = rhs.labels();
    const auto& lbuffer = lhs.tensor().buffer();
    const auto& rbuffer = rhs.tensor().buffer();

    return lbuffer.dot(llabels, rlabels, rbuffer);
}

} // namespace tensorwrapper::tensor::expression
