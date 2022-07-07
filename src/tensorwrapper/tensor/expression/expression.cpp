#include "detail_/add.hpp"
#include "detail_/labeled.hpp"
#include "detail_/pimpl.hpp"
#include "detail_/scale.hpp"
#include "detail_/subtract.hpp"
#include "detail_/times.hpp"

namespace tensorwrapper::tensor::expression {

#define TPARAMS template<typename FieldType>
#define EXPRESSION Expression<FieldType>

// -----------------------------------------------------------------------------
// -- Ctors, Assignment Operators, and Dtor
// -----------------------------------------------------------------------------

TPARAMS
EXPRESSION::Expression() noexcept : Expression(nullptr) {}

TPARAMS
EXPRESSION::Expression(pimpl_pointer p) noexcept : m_pimpl_(std::move(p)) {}

TPARAMS
EXPRESSION::Expression(const Expression& other) :
  Expression(other.m_pimpl_ ? other.m_pimpl_->clone() : nullptr) {}

TPARAMS
EXPRESSION::Expression(Expression&& other) noexcept = default;

TPARAMS
EXPRESSION& EXPRESSION::operator=(const Expression& rhs) {
    if(this != &rhs) Expression(rhs).swap(*this);
    return *this;
}

TPARAMS
EXPRESSION& EXPRESSION::operator=(Expression&& rhs) noexcept = default;

TPARAMS
EXPRESSION::~Expression() noexcept = default;

// -----------------------------------------------------------------------------
// -- Operators
// -----------------------------------------------------------------------------

TPARAMS
EXPRESSION EXPRESSION::operator+(const Expression& rhs) const {
    assert_pimpl_();
    rhs.assert_pimpl_();
    auto pimpl = std::make_unique<detail_::Add<FieldType>>(*this, rhs);
    return Expression(std::move(pimpl));
}

TPARAMS
EXPRESSION EXPRESSION::operator-(const Expression& rhs) const {
    assert_pimpl_();
    rhs.assert_pimpl_();
    auto pimpl = std::make_unique<detail_::Subtract<FieldType>>(*this, rhs);
    return Expression(std::move(pimpl));
}

TPARAMS
EXPRESSION EXPRESSION::operator*(double rhs) const {
    assert_pimpl_();
    auto pimpl = std::make_unique<detail_::Scale<FieldType>>(*this, rhs);
    return Expression(std::move(pimpl));
}

TPARAMS
EXPRESSION EXPRESSION::operator*(const Expression& rhs) const {
    assert_pimpl_();
    rhs.assert_pimpl_();
    auto pimpl = std::make_unique<detail_::Times<FieldType>>(*this, rhs);
    return Expression(std::move(pimpl));
}

// -----------------------------------------------------------------------------
// -- DSL Evaluators
// -----------------------------------------------------------------------------

TPARAMS
typename EXPRESSION::label_type EXPRESSION::labels(
  const_label_reference lhs_labels) const {
    return pimpl_().labels(lhs_labels);
}

TPARAMS
typename EXPRESSION::tensor_type EXPRESSION::tensor(
  const_label_reference labels, const_shape_reference shape,
  const_allocator_reference alloc) const {
    return pimpl_().tensor(labels, shape, alloc);
}

// -----------------------------------------------------------------------------
// -- Utility Methods
// -----------------------------------------------------------------------------

TPARAMS
bool EXPRESSION::is_empty() const noexcept {
    return !static_cast<bool>(m_pimpl_);
}

TPARAMS
void EXPRESSION::swap(Expression& rhs) noexcept { m_pimpl_.swap(rhs.m_pimpl_); }

TPARAMS
bool EXPRESSION::operator==(const Expression& rhs) const noexcept {
    if(m_pimpl_ && rhs.m_pimpl_) return m_pimpl_->are_equal(*rhs.m_pimpl_);
    return is_empty() == rhs.is_empty();
}

TPARAMS
bool EXPRESSION::operator!=(const Expression& rhs) const noexcept {
    return !(*this == rhs);
}

// -----------------------------------------------------------------------------
// -- Private Methods
// -----------------------------------------------------------------------------

TPARAMS
void EXPRESSION::assert_pimpl_() const {
    if(!is_empty()) return;
    throw std::runtime_error("Expression does not contain a PIMPL!!! Was it "
                             "default initialized or moved from?");
}

TPARAMS
typename EXPRESSION::const_pimpl_reference EXPRESSION::pimpl_() const {
    assert_pimpl_();
    return *m_pimpl_;
}

#undef EXPRESSION
#undef TPARAMS

template class Expression<field::Scalar>;
template class Expression<field::Tensor>;

} // namespace tensorwrapper::tensor::expression
