#include "detail_/labeled.hpp"
#include <tensorwrapper/tensor/expressions/labeled_tensor.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expressions {
namespace detail_ {

template<typename FieldType>
struct LabeledTensorPIMPL {
    using my_type     = LabeledTensorPIMPL<FieldType>;
    using parent_type = LabeledTensor<FieldType>;

    auto clone() const { return std::make_unique<my_type>(*this); }

    typename parent_type::label_type m_labels;
    typename parent_type::tensor_type m_tensor;
};

} // namespace detail_

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
                              const_tensor_reference tensor) :
  LabeledTensor(std::make_unique<pimpl_type>(pimpl_type{labels, tensor})) {}

TPARAMS
LABELED_TENSOR::LabeledTensor(const LabeledTensor& rhs) :
  m_pimpl_(rhs.m_pimpl_ ? rhs.m_pimpl_->clone() : nullptr) {}

TPARAMS
LABELED_TENSOR::LabeledTensor(LabeledTensor&& rhs) noexcept = default;

TPARAMS
LABELED_TENSOR& LABELED_TENSOR::operator=(const LabeledTensor& rhs) {
    if(this != &rhs) LABELED_TENSOR(rhs).swap(*this);
    return *this;
}

TPARAMS
LABELED_TENSOR& LABELED_TENSOR::operator=(LabeledTensor&& rhs) noexcept {
    if(this != &rhs) m_pimpl_ = std::move(rhs.m_pimpl_);
    return *this;
}

TPARAMS
LABELED_TENSOR::~LabeledTensor() noexcept = default;

TPARAMS
void LABELED_TENSOR::swap(LabeledTensor& other) noexcept {
    m_pimpl_.swap(other.m_pimpl_);
}

TPARAMS
Expression LABELED_TENSOR::expression() const {
    return Expression(std::make_unique<detail_::Labeled>(*this));
}

TPARAMS
typename LABELED_TENSOR::tensor_reference LABELED_TENSOR::tensor() {
    return pimpl_().m_tensor;
}

TPARAMS
typename LABELED_TENSOR::const_tensor_reference LABELED_TENSOR::tensor() const {
    return pimpl_().m_tensor;
}

TPARAMS
typename LABELED_TENSOR::const_label_reference LABELED_TENSOR::labels() const {
    return pimpl_().m_labels;
}

TPARAMS
LABELED_TENSOR& LABELED_TENSOR::operator=(const Expression& rhs) {
    return *this = std::move(rhs.eval(*this));
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
