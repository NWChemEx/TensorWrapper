#pragma once
#include <tensorwrapper/tensor/fields.hpp>

namespace tensorwrapper::tensor {
template<typename FieldType>
class TensorWrapper;

namespace expression {

/** @brief Associates an annotation with a tensor.
 *
 *
 *  N.B. This is the only part of the expression layer that is templated on the
 *  field. This allows us to catch some errors at compile-time related to
 *  mismatched tensors and ToTs. The rest of the tensor layer is not templated
 *  on this because we need to mix and match tensors and ToTs.
 */
template<typename FieldType>
class LabeledTensor {
public:
    using tensor_type = TensorWrapper<FieldType>;
    using label_type  = typename tensor_type::annotated_type;

    LabeledTensor(label_type labels, tensor_type tensor) noexcept;

    LabeledTensor& operator=(const LabeledTensor& rhs);
    LabeledTensor& operator=(const Expression& rhs);

    Term operator+(const LabeledTensor& rhs);

    const auto& tensor() const { return m_tensor_; }
    const auto& labels() const { return m_labels_; }

private:
    label_type m_labels_;
    tensor_type m_tensor_;
};

extern template LabeledTensor<Field::Scalar>;
extern template LabeledTensor<Field::Tensor>;

} // namespace expression
} // namespace tensorwrapper::tensor
