#include "detail_/add.hpp"
#include "detail_/labeled.hpp"
#include "detail_/scale.hpp"
#include <tensorwrapper/tensor/expressions/expression.hpp>

namespace tensorwrapper::tensor::expressions {

Expression::Expression(pimpl_pointer p) noexcept : m_pimpl_(std::move(p)) {}

Expression::Expression(const Expression& other) :
  Expression(other.m_pimpl_ ? other.m_pimpl_->clone() : nullptr) {}

Expression::Expression(Expression&& other) noexcept = default;

Expression::~Expression() noexcept = default;

Expression Expression::operator+(const Expression& rhs) const {
    auto pimpl = std::make_unique<detail_::Add>(*this, rhs);
    return Expression(std::move(pimpl));
}

Expression Expression::operator*(double rhs) const {
    auto pimpl = std::make_unique<detail_::Scale>(*this, rhs);
    return Expression(std::move(pimpl));
}

Expression Expression::operator*(const Expression& rhs) const {
    throw std::runtime_error("NYI");
    return rhs;
}

typename Expression::labeled_tensor& Expression::eval(
  labeled_tensor& result) const {
    return pimpl_().eval(result);
}

typename Expression::labeled_tot& Expression::eval(labeled_tot& result) const {
    return pimpl_().eval(result);
}

typename Expression::const_pimpl_reference Expression::pimpl_() const {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error("Expression does not contain a PIMPL!!! Was it "
                             "default initialized or moved from?");
}

} // namespace tensorwrapper::tensor::expressions
