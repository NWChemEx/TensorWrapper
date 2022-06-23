#pragma once
#include "pimpl.hpp"
#include <tensorwrapper/tensor/expressions/labeled_tensor.hpp>
#include <variant>

namespace tensorwrapper::tensor::expressions::detail_ {

class Labeled : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using labeled_tensor = typename base_type::labeled_tensor;
    using labeled_tot    = typename base_type::labeled_tot;
    using pimpl_pointer  = typename base_type::pimpl_pointer;

    explicit Labeled(labeled_tensor t) : m_tensor_(std::move(t)) {}
    explicit Labeled(labeled_tot t) : m_tensor_(std::move(t)) {}
    Labeled(const Labeled& other) = default;

protected:
    pimpl_pointer clone_() const override;
    labeled_tensor& eval_(labeled_tensor& lhs) const override;
    labeled_tot& eval_(labeled_tot& lhs) const override;

private:
    bool is_tensor_() const;
    bool is_tot_() const;
    std::variant<labeled_tensor, labeled_tot> m_tensor_;
};

inline typename Labeled::pimpl_pointer Labeled::clone_() const {
    return std::make_unique<Labeled>(*this);
}

inline typename Labeled::labeled_tensor& Labeled::eval_(
  labeled_tensor& result) const {
    if(is_tot_())
        throw std::runtime_error("Error not sure how to convert ToT to tensor");
    const auto& t = std::get<0>(m_tensor_);
    if(t.labels() != result.labels())
        throw std::runtime_error("Permutation NYI");
    return result = t;
}

inline typename Labeled::labeled_tot& Labeled::eval_(
  labeled_tot& result) const {
    if(is_tensor_())
        throw std::runtime_error("Error not sure how to convert tensor to ToT");
    const auto& t = std::get<1>(m_tensor_);
    if(t.labels() != result.labels())
        throw std::runtime_error("Permutation NYI");
    return result = t;
}

inline bool Labeled::is_tensor_() const {
    return std::holds_alternative<labeled_tensor>(m_tensor_);
}

inline bool Labeled::is_tot_() const {
    return std::holds_alternative<labeled_tot>(m_tensor_);
}

} // namespace tensorwrapper::tensor::expressions::detail_
