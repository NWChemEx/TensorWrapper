#pragma once
#include <optional>
#include <tensorwrapper/tensor/expressions/labeled_tensor.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expressions::detail_ {

/** @brief
 *
 *
 *  N.B. Care needs to be taken as whether something is const or not
 *       dramatically changes the behavior. See the function descriptions for
 *       more information.
 */
template<typename FieldType>
struct LabeledTensorPIMPL {
private:
    /// Typedef of this specialization of LabeledTensorPIMPL
    using my_type = LabeledTensorPIMPL<FieldType>;

    /// Typedef of the LabeledTensor specialization this PIMPL implements
    using parent_type = LabeledTensor<FieldType>;

public:
    /// Type used to store the labels (ultimately a typedef of
    /// LabeledTensor::label_type which is string-like)
    using label_type = typename parent_type::label_type;

    /// Type of a read-only reference to the labels (ultimately defined by
    /// LabeledTensor::const_label_reference)
    using const_label_reference = typename parent_type::const_label_reference;

    /// Type of the tensor wrapper (ultimately defined by
    /// LabeledTensor::tensor_type which is TensorWrapper<FieldType>)
    using tensor_type = typename parent_type::tensor_type;

    /// Type of a read-only reference to the tensor (ultimately defined by
    /// LabeledTensor::const_tensor_reference)
    using const_tensor_reference = typename parent_type::const_tensor_reference;

    /// Type of a read/write reference to the tensor (ultimately defined by
    /// LabeledTensor::tensor_reference)
    using tensor_reference = typename parent_type::tensor_reference;

    /// Type of a managed pointer to a LabeledTensorPIMPL instance (ultimately
    /// defined by LabeledTensor::pimpl_pointer resolves to unique_ptr<my_type>)
    using pimpl_pointer = typename parent_type::pimpl_pointer;

    /** @brief Holds empty labels and tensor.
     *
     *  This ctor creates a new LabeledTensorPIMPL instance whose labels are
     *  the empty string and the wrapped tensor is a default instantiated
     *  tensor. The tensor is owned by the LabeledTensorPIMPL instance.
     *
     *  @throws None No throw guarantee.
     */
    LabeledTensorPIMPL() noexcept;

    /** @brief Aliases the provided tensor.
     *
     *  This ctor is selected when labels are applied to a read/write tensor. Of
     *  note this is the ctor that is selected for the labeled tensor that
     *  appears on the left-side of the assignment operator (since it's got to
     *  be writeable for us to assign to it). The resulting LabeledTensorPIMPL
     *  isntance will store a copy of the labels, but only ALIAS the tensor.
     *
     *  Cloning an instance created with this ctor will create a deep-copy and
     *  break the aliasing. Moving will preserve the aliasing.
     *
     *  @param[in] labels The dummy indices labeling the modes of @p tensor
     *  @param[in] tensor The tensor to alias. The user is responsible for
     *                    ensuring @p tensor remains in scope until the created
     *                    LabeldTensorPIMPL instance goes out of scope.
     *
     *  @throws None No throw guarantee.
     */
    LabeledTensorPIMPL(label_type labels, tensor_reference tensor) noexcept;
    LabeledTensorPIMPL(label_type labels, const_tensor_reference tensor);
    LabeledTensorPIMPL(LabeledTensorPIMPL&& other) noexcept;

    pimpl_pointer clone() const;

    const_label_reference labels() const { return m_labels_; }
    tensor_reference tensor() { return *m_ptensor_; }
    const_tensor_reference tensor() const { return *m_ptensor_; }

private:
    /// Deleted to avoid accidental copies
    LabeledTensorPIMPL(const LabeledTensorPIMPL&)            = delete;
    LabeledTensorPIMPL& operator=(const LabeledTensorPIMPL&) = delete;

    label_type m_labels_;
    std::optional<tensor_type> m_tensor_;
    tensor_type* m_ptensor_;
};

} // namespace tensorwrapper::tensor::expressions::detail_

#include "labeled_tensor_pimpl.ipp"
