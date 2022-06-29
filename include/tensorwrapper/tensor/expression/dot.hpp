#pragma once
#include <tensorwrapper/tensor/expression/labeled_view.hpp>

namespace tensorwrapper::tensor::expression {

double dot(const LabeledView<field::Scalar>& lhs,
           const LabeledView<field::Scalar>& rhs);

}
