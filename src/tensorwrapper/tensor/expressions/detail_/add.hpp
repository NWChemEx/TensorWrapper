#pragma once
#include "pimpl.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expressions::detail_ {

template<typename FieldType>
class Add : public ExpressionPIMPL<FieldType> {
private:
    using my_type   = Add<FieldType>;
    using base_type = ExpressionPIMPL<FieldType>;

public:
    using typename base_type::expression_type;
    using typename base_type::labeled_tensor;
    using typename base_type::pimpl_pointer;

    Add(expression_type lhs, expression_type rhs);

protected:
    pimpl_pointer clone_() const override;
    labeled_tensor& eval_(labeled_tensor& lhs) const override;

private:
    expression_type m_lhs_;
    expression_type m_rhs_;
};

template<typename FieldType>
Add<FieldType>::Add(expression_type lhs, expression_type rhs) :
  m_lhs_(std::move(lhs)), m_rhs_(std::move(rhs)) {}

template<typename FieldType>
typename Add<FieldType>::pimpl_pointer Add<FieldType>::clone_() const {
    return std::make_unique<my_type>(*this);
}

template<typename FieldType>
typename Add<FieldType>::labeled_tensor& Add<FieldType>::eval_(
  labeled_tensor& result) const {
    labeled_tensor temp_l(result), temp_r(result);
    temp_l = m_lhs_.eval(temp_l);
    temp_r = m_rhs_.eval(temp_r);

    const auto& result_labels = result.labels();
    const auto& l_labels      = temp_l.labels();
    const auto& r_labels      = temp_r.labels();

    auto& result_buffer = result.tensor().buffer();
    const auto& lbuffer = temp_l.tensor().buffer();
    const auto& rbuffer = temp_r.tensor().buffer();

    lbuffer.add(l_labels, result_labels, result_buffer, r_labels, rbuffer);

    return result;
}

} // namespace tensorwrapper::tensor::expressions::detail_
