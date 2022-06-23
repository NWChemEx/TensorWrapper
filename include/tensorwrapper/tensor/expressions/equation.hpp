#pragma once

namespace tensorwrapper::tensor::expressions {

template<typename FieldType>
class Equation {
    using labeled_tensor      = LabeledTensor<FieldType>;
    using tensor_wrapper_type = typename labeled_tensor::tensor_type;
    using expression_type     = Expression;

    Equation(labeled_tensor t, expression_type e);

    tensor_wrapper_type eval();

private:
    LabeledTensor<FieldType> m_lhs_;

    Expression m_rhs_;
};

template<typename FieldType>
Equation<FieldType>::Equation(labeled_tensor t, expression_type e) :
  m_result_(std::move(t)), m_lhs_(std::move(e)) {}

template<typename FieldType>
typename Equation<FieldType>::tensor_wrapper_type eval() {
    return m_rhs_.eval(m_result_);
}

} // namespace tensorwrapper::tensor::expressions
