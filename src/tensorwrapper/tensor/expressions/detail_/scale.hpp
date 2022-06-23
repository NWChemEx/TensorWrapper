#pragma once
#include "pimpl.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expressions::detail_ {

class Scale : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using expression_type = typename base_type::expression_type;
    using labeled_tensor  = typename base_type::labeled_tensor;
    using labeled_tot     = typename base_type::labeled_tot;
    using pimpl_pointer   = typename base_type::pimpl_pointer;

    Scale(expression_type lhs, double rhs);
    Scale(const Scale& other) = default;

protected:
    pimpl_pointer clone_() const override;
    labeled_tensor& eval_(labeled_tensor& result) const override;
    labeled_tot& eval_(labeled_tot& result) const override;

private:
    template<typename T>
    T& eval_common_(T& result) const;

    expression_type m_lhs_;
    double m_rhs_;
};

inline Scale::Scale(expression_type lhs, double rhs) :
  m_lhs_(std::move(lhs)), m_rhs_(rhs) {}

inline typename Scale::pimpl_pointer Scale::clone_() const {
    return std::make_unique<Scale>(*this);
}

inline typename Scale::labeled_tensor& Scale::eval_(
  labeled_tensor& result) const {
    return eval_common_(result);
}

inline typename Scale::labeled_tot& Scale::eval_(labeled_tot& result) const {
    return eval_common_(result);
}

template<typename T>
T& Scale::eval_common_(T& result) const {
    T temp(result);
    temp = m_lhs_.eval(temp);

    const auto& rlabels = temp.labels();
    const auto& llabels = result.labels();

    auto& rbuffer = temp.tensor().buffer();
    auto& lbuffer = result.tensor().buffer();

    rbuffer.scale(rlabels, llabels, lbuffer, m_rhs_);

    return result;
}

} // namespace tensorwrapper::tensor::expressions::detail_
