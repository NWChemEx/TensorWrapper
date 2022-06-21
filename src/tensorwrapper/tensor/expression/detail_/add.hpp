#pragma once
#include "pimpl.hpp"

namespace tensorwrapper::tensor::expression::detail_ {

class Add : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using expression_type = typename base_type::expression_type;
    using labeled_tensor  = typename base_type::labeled_tensor;
    using labeled_tot     = typename base_type::labeled_tot;

    Add(expression_type lhs, expression_type rhs);

protected:
    labeled_tensor eval_(const labeled_tensor& result) const override;
    labeled_tot eval_(const labeled_tot& result) const override;

private:
    template<typename T>
    T eval_common_(const T& result) const;

    expression_type m_lhs_;
    expression_type m_rhs_;
};

inline Add::Add(expression_type lhs, expression_type rhs) :
  m_lhs_(std::move(lhs)), m_rhs_(std::move(rhs)) {}

inline typename Add::labeled_tensor Add::eval_(
  const labeled_tensor& result) const {
    return eval_common_(result);
}

inline typename Add::labeled_tot Add::eval_(const labeled_tot& result) const {
    return eval_common_(result);
}

template<typename T>
T Add::eval_common_(const T& result) const {
    auto l = m_lhs_.eval(result);
    auto r = m_rhs_.eval(result);

    T rv(result);
    auto& result_buffer = rv.tensor().buffer();
    const auto& lbuffer = l.tensor().buffer();
    const auto& rbuffer = r.tensor().buffer();

    lbuffer.add(l.labels(), rv.labels(), result_buffer, r.labels(), rbuffer);

    return rv;
}

} // namespace tensorwrapper::tensor::expression::detail_
