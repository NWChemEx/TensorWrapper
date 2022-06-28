#include "detail_/labeled.hpp"
#include <tensorwrapper/tensor/expressions/labeled_view.hpp>
namespace tensorwrapper::tensor::expressions {

#define TPARAMS template<typename FieldType>
#define LABELED_VIEW LabeledView<FieldType>

// -----------------------------------------------------------------------------
// -- ctors, assignment operators, and dtor
// -----------------------------------------------------------------------------

TPARAMS
LABELED_VIEW::LabeledView(label_type labels, tensor_reference tensor) noexcept :
  m_labels_(std::move(labels)), m_tensor_(std::ref(tensor)), m_ctensor_() {}

TPARAMS LABELED_VIEW::LabeledView(label_type labels,
                                  const_tensor_reference tensor) noexcept :
  m_labels_(std::move(labels)), m_tensor_(), m_ctensor_(std::cref(tensor)) {}

TPARAMS
Expression<FieldType> LABELED_VIEW::expression() const {
    using labeled_type = detail_::LabeledType<FieldType>;
    return Expression<FieldType>(std::make_unique<labeled_type>(*this));
}

TPARAMS
typename LABELED_VIEW::tensor_reference LABELED_VIEW::tensor() {
    if(m_tensor_) return *m_tensor_;
    throw std::runtime_error("Not holding a read/write tensor");
}

TPARAMS
typename LABELED_VIEW::const_tensor_reference LABELED_VIEW::tensor() const {
    if(m_tensor_)
        return *m_tensor_;
    else if(m_ctensor_)
        return *m_ctensor_;
    throw std::runtime_error("Not holding a tensor");
}

TPARAMS
LABELED_VIEW& LABELED_VIEW::operator=(const LabeledView& rhs) {
    return operator=(rhs.expression());
}

TPARAMS
LABELED_VIEW& LABELED_VIEW::operator=(const Expression<FieldType>& rhs) {
    tensor_type rv(rhs.shape(*this), rhs.allocator(*this));

      auto& rv = rhs.eval(*this);
    if(&rv != this)
        throw std::runtime_error("Expected to get result back by reference");
    return *this;
}

// TPARAMS
// Expression<FieldType> LABELED_VIEW::operator+(const LabeledView& rhs) const {
//     return expression() + rhs.expression();
// }

// TPARAMS
// Expression LABELED_VIEW::operator*(double rhs) const {
//     return expression() * rhs;
// }

#undef LABELED_VIEW
#undef TPARAMS

template class LabeledView<field::Scalar>;
template class LabeledView<field::Tensor>;

} // namespace tensorwrapper::tensor::expressions
