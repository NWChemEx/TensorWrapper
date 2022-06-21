#pragma once
#include "pimpl.hpp"
#include <variant>

namespace tensorwrapper::tensor::expression::detail_ {

class Labeled : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using labeled_tensor = typename base_type::labeled_tensor;
    using labeled_tot    = typename base_type::labeled_tot;

    explicit Labeled(labeled_tensor t) : m_tensor_(std::move(t)) {}
    explicit Labeled(labeled_tot t) : m_tensor_(std::move(t)) {}

protected:
    labeled_tensor eval_(const labeled_tensor& lhs) const override;
    labeled_tot eval_(const labeled_tot& lhs) const override;

private:
    bool is_tensor_() const;
    bool is_tot_() const;
    std::variant<labeled_tensor, labeled_tot> m_tensor_;
};

inline typename Labeled::label_type Labeled::output_labels_(
  const label_type& lhs) const {
    return lhs;
}

inline typename Labeled::labeld_tensor Labeled::eval_(
  const labeled_tensor& result) const {
    if(is_tot_())
        throw std::runtime_error("Error not sure how to convert ToT to tensor");
    const auto& t = std::get<0>(m_tensor_);
    if(t.labels() != result.labels())
        throw std::runtime_error("Permutation NYI");
    return t;
}

inline typename Labeled::labeled_tot Labeled::eval_(
  const labeled_tot& lhs) const {
    if(is_tensor_())
        throw std::runtime_error("Error not sure how to convert tensor to ToT");
    const auto& t = std::get<1>(m_tensor_);
    if(t.labels() != result.labels())
        throw std::runtime_error("Permutation NYI");
    return t;
}

inline bool Labeled::is_tensor_() const {
    return std::holds_alternative<labeled_tensor>(m_tensor_);
}

inline bool Labeled::is_tot_() const {
    return std::holds_alternative<labeled_tot>(m_tensor_);
}

} // namespace tensorwrapper::tensor::expression::detail_
