#include "detail_/add.hpp"
#include "detail_/labeled.hpp"
#include "detail_/scale.hpp"
#include "detail_/subtract.hpp"
#include <tensorwrapper/tensor/expression/expression_class.hpp>

namespace tensorwrapper::tensor::expression {

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

TPARAMS
EXPRESSION EXPRESSION::operator+(const Expression& rhs) const {
    auto pimpl = std::make_unique<detail_::Add<FieldType>>(*this, rhs);
    return Expression(std::move(pimpl));
}

TPARAMS
EXPRESSION EXPRESSION::operator-(const Expression& rhs) const {
    auto pimpl = std::make_unique<detail_::Subtract<FieldType>>(*this, rhs);
    return Expression(std::move(pimpl));
}

TPARAMS
EXPRESSION EXPRESSION::operator*(double rhs) const {
    auto pimpl = std::make_unique<detail_::Scale<FieldType>>(*this, rhs);
    return Expression(std::move(pimpl));
}

TPARAMS
EXPRESSION EXPRESSION::operator*(const Expression& rhs) const {
    throw std::runtime_error("NYI");
    return rhs;
}

TPARAMS
typename EXPRESSION::tensor_type EXPRESSION::tensor(
  const_label_reference labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    return pimpl_().tensor(labels, shape, alloc);
}

TPARAMS
bool EXPRESSION::operator==(const Expression& rhs) const noexcept {
    if(m_pimpl_ && rhs.m_pimpl_) return m_pimpl_->are_equal(*rhs.m_pimpl_);
    return static_cast<bool>(m_pimpl_) == static_cast<bool>(rhs.m_pimpl_);
}

TPARAMS
bool EXPRESSION::operator!=(const Expression& rhs) const noexcept {
    return !(*this == rhs);
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

} // namespace tensorwrapper::tensor::expression
