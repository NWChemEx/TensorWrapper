#pragma once
#include "pimpl.hpp"
#include <tensorwrapper/tensor/expressions/labeled_view.hpp>
#include <variant>

namespace tensorwrapper::tensor::expressions::detail_ {

template<typename FieldType>
class Labeled : public ExpressionPIMPL<FieldType> {
private:
    using my_type   = Labeled<FieldType>;
    using base_type = ExpressionPIMPL<FieldType>;

public:
    using labeled_tensor = typename base_type::labeled_tensor;
    using pimpl_pointer  = typename base_type::pimpl_pointer;

    explicit Labeled(labeled_tensor t) : m_tensor_(std::move(t)) {}
    Labeled(const Labeled& other) = default;

protected:
    pimpl_pointer clone_() const override;
    labeled_tensor& eval_(labeled_tensor& lhs) const override;

private:
    labeled_tensor m_tensor_;
};

template<typename FieldType>
typename Labeled<FieldType>::pimpl_pointer Labeled<FieldType>::clone_() const {
    return std::make_unique<my_type>(*this);
}

template<typename FieldType>
typename Labeled<FieldType>::labeled_tensor& Labeled<FieldType>::eval_(
  labeled_tensor& result) const {
    const auto& rhs_labels = m_tensor_.labels();
    const auto& lhs_labels = result.labels();
    const auto& rhs_buffer = m_tensor_.tensor().buffer();
    auto& lhs_buffer       = result.tensor().buffer();

    if(rhs_labels != lhs_labels) {
        rhs_buffer.permute(rhs_labels, lhs_labels, lhs_buffer);
    } else {
        lhs_buffer = rhs_buffer;
    }

    return result;
}

} // namespace tensorwrapper::tensor::expressions::detail_
