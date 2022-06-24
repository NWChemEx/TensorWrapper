// This file meant only for inclusion in labeled_tensor_pimpl.hpp

namespace tensorwrapper::tensor::expressions::detail_ {

#define TPARAMS template<typename FieldType>
#define LABELED_TENSOR_PIMPL LabeledTensorPIMPL<FieldType>

TPARAMS
LABELED_TENSOR_PIMPL::LabeledTensorPIMPL() noexcept :
  LabeledTensorPIMPL("", tensor_type{}) {}

TPARAMS
LABELED_TENSOR_PIMPL::LabeledTensorPIMPL(label_type labels,
                                         tensor_reference t) noexcept :
  m_labels_(std::move(labels)), m_tensor_(), m_ptensor_(&t) {}

TPARAMS
LABELED_TENSOR_PIMPL::LabeledTensorPIMPL(label_type labels,
                                         const_tensor_reference t) :
  m_labels_(std::move(labels)), m_tensor_(t), m_ptensor_(&m_tensor_.value()) {}

TPARAMS
LABELED_TENSOR_PIMPL::LabeledTensorPIMPL(LabeledTensorPIMPL&& other) noexcept :
  m_labels_(std::move(other.m_labels_)),
  m_tensor_(std::move(other.m_tensor_)),
  m_ptensor_(m_tensor_ ? &m_tensor_.value() : other.m_ptensor_) {}

TPARAMS
typename LABELED_TENSOR_PIMPL::pimpl_pointer LABELED_TENSOR_PIMPL::clone()
  const {
    const auto& the_labels = m_labels_;
    const auto& the_tensor = *m_ptensor_;
    return std::make_unique<my_type>(the_labels, the_tensor);
}

#undef LABELED_TENSOR_PIMPL
#undef TPARAMS

} // namespace tensorwrapper::tensor::expressions::detail_
