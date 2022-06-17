#include <tensorwrapper/tensor/expression/labeled_tensor.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expression {

#define TPARAMS template<typename FieldType>
#define LABELED_TENSOR LabeledTensor<FieldType>

TPARAMS
LABELED_TENSOR::LabeledTensor(label_type labels, tensor_type tensor) noexcept :
  m_labels_(std::move(labels)), m_tensor_(std::move(tensor)) {}

TPARAMS
LABELED_TENSOR& LABELED_TENSOR::operator=(const LabeledTensor& rhs) {
    return *this = Expression(rhs);
}

TPARAMS
LABELED_TENSOR& LABELED_TENSOR::operator=(const Expression& rhs) {
    m_tensor_ = rhs.eval(*this);
    return *this;
}

#undef LABELED_TENSOR
#undef TPARAMS

template LabeledTensor<Field::Scalar>;
template LabeledTensor<Field::Tensor>;

} // namespace tensorwrapper::tensor::expression
