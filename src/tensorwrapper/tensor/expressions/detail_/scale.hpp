#pragma once
#include "pimpl.hpp"
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expressions::detail_ {

template<typename FieldType>
class Scale : public ExpressionPIMPL<FieldType> {
private:
    using base_type = ExpressionPIMPL<FieldType>;

public:
    using expression_type = typename base_type::expression_type;
    using labeled_tensor  = typename base_type::labeled_tensor;
    using pimpl_pointer   = typename base_type::pimpl_pointer;

    Scale(expression_type lhs, double rhs);
    Scale(const Scale& other) = default;

protected:
    pimpl_pointer clone_() const override;
    labeled_tensor& eval_(labeled_tensor& result) const override;

private:
    expression_type m_lhs_;
    double m_rhs_;
};

template<typename FieldType>
Scale<FieldType>::Scale(expression_type lhs, double rhs) :
  m_lhs_(std::move(lhs)), m_rhs_(rhs) {}

template<typename FieldType>
typename Scale<FieldType>::pimpl_pointer Scale<FieldType>::clone_() const {
    return std::make_unique<Scale>(*this);
}

template<typename FieldType>
typename Scale<FieldType>::labeled_tensor& Scale<FieldType>::eval_(
  labeled_tensor& result) const {
    labeled_tensor temp(result);
    temp = m_lhs_.eval(temp);

    const auto& rlabels = temp.labels();
    const auto& llabels = result.labels();

    auto& rbuffer = temp.tensor().buffer();
    auto& lbuffer = result.tensor().buffer();

    rbuffer.scale(rlabels, llabels, lbuffer, m_rhs_);

    return result;
}

} // namespace tensorwrapper::tensor::expressions::detail_
