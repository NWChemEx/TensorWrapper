#pragma once
#include "pimpl.hpp"

namespace tensorwrapper::tensor::expression::detail_ {

class Add : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using expression_type = typename base_type::expression_type;
    using labeled_tensor  = typename base_type::labeled_tensor;
    using tensor_type     = typename base_type::tensor_type;
    using labeled_tot     = typename base_type::labeled_tot;
    using tot_type        = typename base_type::tot_type;

    Add(expression_type lhs, expression_type rhs);

protected:
    tensor_type eval_(const labeled_tensor& lhs) const override;
    tot_type eval_(const labeled_tot& lhs) const override;

private:
    expression_type m_lhs_;
    expression_type m_rhs_;
};

inline Add::Add(expression_type lhs, expression_type rhs) :
  m_lhs_(std::move(lhs)), m_rhs_(std::move(rhs)) {}

inline typename Add::tensor_type Add::eval_(const labeled_tensor& lhs) const {
    auto result_labels = lhs.labels();
    auto l             = m_lhs_.eval(lhs);
    auto r             = m_rhs_.eval(rhs);
    l.tensor().buffer().add();

} // namespace tensorwrapper::tensor::expression::detail_
