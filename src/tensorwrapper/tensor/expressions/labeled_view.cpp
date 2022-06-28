#include "detail_/labeled.hpp"
#include <tensorwrapper/tensor/allocators/allocators.hpp>
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
typename LABELED_VIEW::expression_type LABELED_VIEW::expression() const {
    using labeled_type = detail_::Labeled<FieldType>;
    return expression_type(std::make_unique<labeled_type>(*this));
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
LABELED_VIEW& LABELED_VIEW::operator=(const_expression_reference rhs) {
    // TODO: shape and alloc should come from rhs and used to initialize
    //       the shape and allocator for this instance (unless the user provides
    //       the shape and/or the allocator)
    //
    // auto alloc = rhs.allocator();
    // auto shape = rhs.shape(*this);
    // TensorWrapper(std::move(shape), std::move(alloc)).swap(*m_tensor_);
    // Now call eval to fill in buffer
    rhs.eval(*this);

    // As a hack we fill in the buffer first and then use the buffer to create
    // the shape. We just assume the default allocator for now
    using shape_type = typename tensor_type::shape_type;

    auto& buffer = tensor().buffer();
    auto shape   = std::make_unique<shape_type>(buffer.make_extents());
    auto alloc   = default_allocator<FieldType>();

    TensorWrapper(std::move(buffer), std::move(shape), std::move(alloc))
      .swap(*m_tensor_);
    return *this;
}

TPARAMS
typename LABELED_VIEW::expression_type LABELED_VIEW::operator+(
  const LabeledView& rhs) const {
    return expression() + rhs.expression();
}

TPARAMS
typename LABELED_VIEW::expression_type LABELED_VIEW::operator*(
  double rhs) const {
    return expression() * rhs;
}

#undef LABELED_VIEW
#undef TPARAMS

template class LabeledView<field::Scalar>;
template class LabeledView<field::Tensor>;

} // namespace tensorwrapper::tensor::expressions
