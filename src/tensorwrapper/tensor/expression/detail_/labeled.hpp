#pragma once
#include "pimpl.hpp"
#include <variant>

namespace tensorwrapper::tensor::expression::detail_ {

class Labeled : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using labeled_tensor = typename base_type::labeled_tensor;
    using tensor_type    = typename base_type::tensor_type;
    using labeled_tot    = typename base_type::labeled_tot;
    usint tot_type       = typename base_type::tot_type;

    explicit Labeled(labeled_tensor t) : m_tensor_(std::move(t)) {}
    explicit Labeled(labeled_tot t) : m_tensor_(std::move(t)) {}

protected:
    label_type output_labels_(const label_type& lhs) const override;
    tensor_type eval_(const labeled_tensor& lhs) const override;
    tot_type eval_(const labeled_tot& lhs) const override;

private:
    bool is_tensor_() const;
    bool is_tot_() const;
    std::variant<labeled_tensor, labeled_tot> m_tensor_;
};

inline typename Labeled::label_type Labeled::output_labels_(
  const label_type& lhs) const {
    return lhs;
}

inline typename Labeled::tensor_type Labeled::eval_(
  const labeled_tensor& lhs) const {
    if(is_tot_())
        throw std::runtime_error("Error not sure how to convert ToT to tensor");
    const auto& t = std::get<0>(m_tensor_);
    if(t.labels() != lhs.labels()) throw std::runtime_error("Permutation NYI");
    return tensor_type(t.tensor());
}

inline typename Labeled::tot_type Labeled::eval_(const labeled_tot& lhs) const {
    if(is_tensor_())
        throw std::runtime_error("Error not sure how to convert tensor to ToT");
    const auto& t = std::get<1>(m_tensor_);
    if(t.labels() != lhs.labels()) throw std::runtime_error("Permutation NYI");
    return tot_type(t.tensor());
}

inline bool Labeled::is_tensor_() const {
    return std::holds_alternative<labeled_tensor>(m_tensor_);
}

inline bool Labeled::is_tot_() const {
    return std::holds_alternative<labeled_tot>(m_tensor_);
}

} // namespace tensorwrapper::tensor::expression::detail_
