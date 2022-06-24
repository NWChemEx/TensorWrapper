#pragma once
#include <memory>
#include <tensorwrapper/tensor/expressions/expression.hpp>
#include <tensorwrapper/tensor/fields.hpp>

namespace tensorwrapper::tensor {
template<typename FieldType>
class TensorWrapper;

namespace expressions {
namespace detail_ {
template<typename FieldType>
class LabeledTensorPIMPL;

}

class Expression;

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
    using tensor_type            = TensorWrapper<FieldType>;
    using tensor_reference       = tensor_type&;
    using const_tensor_reference = const tensor_type&;
    using label_type             = std::string;
    using const_label_reference  = const label_type&;
    using pimpl_type             = detail_::LabeledTensorPIMPL<FieldType>;
    using pimpl_pointer          = std::unique_ptr<pimpl_type>;

    explicit LabeledTensor(pimpl_pointer p = nullptr) noexcept;
    LabeledTensor(const_label_reference labels, tensor_reference tensor);
    LabeledTensor(const_label_reference labels, const_tensor_reference tensor);
    LabeledTensor(const LabeledTensor& other);
    // LabeledTensor(LabeledTensor&& other) noexcept;
    ~LabeledTensor() noexcept;

    void swap(LabeledTensor& other) noexcept;

    Expression expression() const;
    tensor_reference tensor();
    const_tensor_reference tensor() const;
    const_label_reference labels() const;

    LabeledTensor& operator=(const LabeledTensor& rhs);
    LabeledTensor& operator=(const Expression& rhs);
    Expression operator+(const LabeledTensor& rhs) const;
    Expression operator*(double rhs) const;

private:
    // LabeledTensor& operator=(LabeledTensor&&) = delete;

    pimpl_type& pimpl_();
    const pimpl_type& pimpl_() const;
    pimpl_pointer m_pimpl_;
};

template<typename FieldType>
Expression operator*(double lhs, const LabeledTensor<FieldType>& rhs) {
    return rhs * lhs;
}

extern template class LabeledTensor<field::Scalar>;
extern template class LabeledTensor<field::Tensor>;

} // namespace expressions
} // namespace tensorwrapper::tensor
