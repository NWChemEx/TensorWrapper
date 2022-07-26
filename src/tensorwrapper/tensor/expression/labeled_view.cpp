#include "detail_/labeled.hpp"
#include <tensorwrapper/tensor/allocator/allocator_class.hpp>
#include <tensorwrapper/tensor/expression/labeled_view.hpp>
namespace tensorwrapper::tensor::expression {

#define TPARAMS template<typename FieldType>
#define LABELED_VIEW LabeledView<FieldType>

// -----------------------------------------------------------------------------
// -- Ctors, assignment operators, and dtor
// -----------------------------------------------------------------------------

TPARAMS
LABELED_VIEW::LabeledView(label_type labels, tensor_reference tensor) noexcept :
  m_labels_(std::move(labels)), m_tensor_(std::ref(tensor)), m_ctensor_() {}

TPARAMS LABELED_VIEW::LabeledView(label_type labels,
                                  const_tensor_reference tensor) noexcept :
  m_labels_(std::move(labels)), m_tensor_(), m_ctensor_(std::cref(tensor)) {}

// -----------------------------------------------------------------------------
// -- Accessors
// -----------------------------------------------------------------------------

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

// -----------------------------------------------------------------------------
// -- Assignment operations
// -----------------------------------------------------------------------------

TPARAMS
LABELED_VIEW& LABELED_VIEW::operator=(const LabeledView& rhs) {
    return operator=(rhs.expression());
}

TPARAMS
LABELED_VIEW& LABELED_VIEW::operator=(const_expression_reference rhs) {
    if(!m_tensor_)
        throw std::runtime_error("Can not assign to a read-only tensor");

    // TODO: shape and alloc should come from rhs and used to initialize
    //       the shape and allocator for this instance (unless the user provides
    //       the shape and/or the allocator)
    //
    // auto alloc = rhs.allocator();
    // auto shape = rhs.shape(*this);
    // TensorWrapper(std::move(shape), std::move(alloc)).swap(*m_tensor_);
    // Now call tensor to make it
    // return rhs.tensor(labels(), tensor().shape(), tensor().allocator());

    // As a hack we use the default allocator and an empty shape to get the
    // correct buffer. We then use the buffer to get the correct shape

    using shape_type = typename tensor_type::shape_type;
    auto alloc       = default_allocator<FieldType>();
    auto temp        = rhs.tensor(labels(), shape_type(), *alloc);

    auto& buffer       = temp.buffer();
    auto outer_extents = buffer.make_extents();
    auto inner_extents = buffer.make_inner_extents();
    auto shape = std::make_unique<shape_type>(outer_extents, inner_extents);

    tensor() =
      TensorWrapper(std::move(buffer), std::move(shape), std::move(alloc));
    return *this;
}

// -----------------------------------------------------------------------------
// -- Math operations
// -----------------------------------------------------------------------------

TPARAMS
typename LABELED_VIEW::expression_type LABELED_VIEW::operator+(
  const LabeledView& rhs) const {
    return expression() + rhs.expression();
}

TPARAMS
typename LABELED_VIEW::expression_type LABELED_VIEW::operator-(
  const LabeledView& rhs) const {
    return expression() - rhs.expression();
}

TPARAMS
typename LABELED_VIEW::expression_type LABELED_VIEW::operator*(
  const LabeledView& rhs) const {
    return expression() * rhs.expression();
}

TPARAMS
typename LABELED_VIEW::expression_type LABELED_VIEW::operator*(
  double rhs) const {
    return expression() * rhs;
}

// -----------------------------------------------------------------------------
// -- Utility
// -----------------------------------------------------------------------------

TPARAMS
bool LABELED_VIEW::operator==(const LabeledView& rhs) const noexcept {
    // Check if they both have or both don't have a read/write tensor
    if(static_cast<bool>(m_tensor_) != static_cast<bool>(rhs.m_tensor_))
        return false;

    // Check if they both have or both don't have a read-only tensor
    if(static_cast<bool>(m_ctensor_) != static_cast<bool>(rhs.m_ctensor_))
        return false;

    // Make sure *this has a tensor (if not they both don't have one)
    if(!(m_tensor_ || m_ctensor_)) return true;

    // These won't throw because they both have tensors
    const auto* ptensor     = &tensor();
    const auto* rhs_ptensor = &rhs.tensor();

    // Now compare addresses of tensors and labels
    return std::tie(ptensor, m_labels_) == std::tie(rhs_ptensor, rhs.m_labels_);
}

TPARAMS
bool LABELED_VIEW::operator!=(const LabeledView& rhs) const noexcept {
    return !(*this == rhs);
}

#undef LABELED_VIEW
#undef TPARAMS

template class LabeledView<field::Scalar>;
template class LabeledView<field::Tensor>;

} // namespace tensorwrapper::tensor::expression
