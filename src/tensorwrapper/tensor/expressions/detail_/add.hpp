#pragma once
#include "pimpl.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expressions::detail_ {

class Add : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using expression_type = typename base_type::expression_type;
    using labeled_tensor  = typename base_type::labeled_tensor;
    using labeled_tot     = typename base_type::labeled_tot;
    using pimpl_pointer   = typename base_type::pimpl_pointer;

    Add(expression_type lhs, expression_type rhs);
    Add(const Add& other) = default;

protected:
    pimpl_pointer clone_() const override;
    labeled_tensor& eval_(labeled_tensor& result) const override;
    labeled_tot& eval_(labeled_tot& result) const override;

private:
    template<typename T>
    T& eval_common_(T& result) const;

    expression_type m_lhs_;
    expression_type m_rhs_;
};

inline Add::Add(expression_type lhs, expression_type rhs) :
  m_lhs_(std::move(lhs)), m_rhs_(std::move(rhs)) {}

inline typename Add::pimpl_pointer Add::clone_() const {
    return std::make_unique<Add>(*this);
}

inline typename Add::labeled_tensor& Add::eval_(labeled_tensor& result) const {
    return eval_common_(result);
}

inline typename Add::labeled_tot& Add::eval_(labeled_tot& result) const {
    return eval_common_(result);
}

template<typename T>
T& Add::eval_common_(T& result) const {
    T temp_l(result), temp_r(result);
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
