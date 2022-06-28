#pragma once
#include <tensorwrapper/tensor/expressions/labeled_view.hpp>

namespace tensorwrapper::tensor::expressions {

double dot(const LabeledView<field::Scalar>& lhs,
           const LabeledView<field::Scalar>& rhs);

}
