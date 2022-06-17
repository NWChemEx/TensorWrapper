#pragma once

namespace tensorwrapper::tensor::expression {

template<typename FieldType>
class Equation {
private:
    LabeledTensor<FieldType> m_result_;
    Expression m_rhs_;
};

} // namespace tensorwrapper::tensor::expression
