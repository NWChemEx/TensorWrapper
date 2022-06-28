#include "detail_/add.hpp"
#include "detail_/labeled.hpp"
#include "detail_/scale.hpp"
#include <tensorwrapper/tensor/expressions/expression.hpp>

namespace tensorwrapper::tensor::expressions {

#define TPARAMS template<typename FieldType>
#define EXPRESSION Expression<FieldType>

TPARAMS
EXPRESSION::Expression(pimpl_pointer p) noexcept : m_pimpl_(std::move(p)) {}

TPARAMS
EXPRESSION::Expression(const Expression& other) :
  Expression(other.m_pimpl_ ? other.m_pimpl_->clone() : nullptr) {}

TPARAMS
EXPRESSION::Expression(Expression&& other) noexcept = default;

TPARAMS
EXPRESSION::~Expression() noexcept = default;

/*
EXPRESSION Expression::operator+(const Expression& rhs) const {
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
*/

TPARAMS
typename EXPRESSION::labeled_tensor& EXPRESSION::eval(
  labeled_tensor& result) const {
    return pimpl_().eval(result);
}

TPARAMS
typename EXPRESSION::const_pimpl_reference EXPRESSION::pimpl_() const {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error("Expression does not contain a PIMPL!!! Was it "
                             "default initialized or moved from?");
}

#undef EXPRESSION
#undef TPARAMS

template class Expression<field::Scalar>;
template class Expression<field::Tensor>;

} // namespace tensorwrapper::tensor::expressions
