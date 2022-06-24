#include "detail_/labeled.hpp"
#include "detail_/labeled_tensor_pimpl.hpp"
namespace tensorwrapper::tensor::expressions {

#define TPARAMS template<typename FieldType>
#define LABELED_TENSOR LabeledTensor<FieldType>

// -----------------------------------------------------------------------------
// -- ctors, assignment operators, and dtor
// -----------------------------------------------------------------------------

TPARAMS
LABELED_TENSOR::LabeledTensor(pimpl_pointer p) noexcept :
  m_pimpl_(std::move(p)) {}

TPARAMS
LABELED_TENSOR::LabeledTensor(const_label_reference labels,
                              tensor_reference tensor) :
  LabeledTensor(std::make_unique<pimpl_type>(labels, tensor)) {}

TPARAMS
LABELED_TENSOR::LabeledTensor(const_label_reference labels,
                              const_tensor_reference tensor) :
  LabeledTensor(std::make_unique<pimpl_type>(labels, tensor)) {}

TPARAMS
LABELED_TENSOR::LabeledTensor(const LabeledTensor& other) :
  LabeledTensor(other.m_pimpl_ ? other.m_pimpl_->clone() : nullptr) {}

// TPARAMS
// LABELED_TENSOR::LabeledTensor(LabeledTensor&& other) noexcept = default;

TPARAMS
LABELED_TENSOR::~LabeledTensor() noexcept = default;

TPARAMS
void LABELED_TENSOR::swap(LabeledTensor& other) noexcept {
    if(this != &other) m_pimpl_.swap(other.m_pimpl_);
}

TPARAMS
Expression LABELED_TENSOR::expression() const {
    return Expression(std::make_unique<detail_::Labeled>(*this));
}

TPARAMS
typename LABELED_TENSOR::tensor_reference LABELED_TENSOR::tensor() {
    return pimpl_().tensor();
}

TPARAMS
typename LABELED_TENSOR::const_tensor_reference LABELED_TENSOR::tensor() const {
    return pimpl_().tensor();
}

TPARAMS
typename LABELED_TENSOR::const_label_reference LABELED_TENSOR::labels() const {
    return pimpl_().labels();
}

TPARAMS
LABELED_TENSOR& LABELED_TENSOR::operator=(const LabeledTensor& rhs) {
    return operator=(rhs.expression());
}

TPARAMS
LABELED_TENSOR& LABELED_TENSOR::operator=(const Expression& rhs) {
    auto& rv = rhs.eval(*this);
    if(&rv != this)
        throw std::runtime_error("Expected to get result back by reference");
    return *this;
}

TPARAMS
Expression LABELED_TENSOR::operator+(const LabeledTensor& rhs) const {
    return expression() + rhs.expression();
}

TPARAMS
Expression LABELED_TENSOR::operator*(double rhs) const {
    return expression() * rhs;
}

TPARAMS
typename LABELED_TENSOR::pimpl_type& LABELED_TENSOR::pimpl_() {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error("LabeledTensor has no PIMPL!!! Was it default "
                             "constructed and/or moved from?");
}

TPARAMS
const typename LABELED_TENSOR::pimpl_type& LABELED_TENSOR::pimpl_() const {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error("LabeledTensor has no PIMPL!!! Was it default "
                             "constructed and/or moved from?");
}

#undef LABELED_TENSOR
#undef TPARAMS

template class LabeledTensor<field::Scalar>;
template class LabeledTensor<field::Tensor>;

} // namespace tensorwrapper::tensor::expressions
