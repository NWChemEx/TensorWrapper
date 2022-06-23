#pragma once
#include <tensorwrapper/tensor/expressions/labeled_tensor.hpp>

namespace tensorwrapper::tensor::expressions {

double dot(const LabeledTensor<field::Scalar>& lhs,
           const LabeledTensor<field::Scalar>& rhs);

}
