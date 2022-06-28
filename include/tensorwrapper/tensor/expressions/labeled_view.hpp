#pragma once
#include <memory>
#include <optional>
#include <tensorwrapper/tensor/expressions/expression.hpp>
#include <tensorwrapper/tensor/fields.hpp>

namespace tensorwrapper::tensor {
template<typename FieldType>
class TensorWrapper;

namespace expressions {

template<typename FieldType>
class Expression;

/** @brief Associates an annotation with a reference to a tensor.
 *
 *  This class associates a set of labels with a reference to a tensor. In
 *  particular this means the LabeledView instance does not manage the lifetime
 *  of the TensorWrapper used to create it and it is the caller's responsibility
 *  to ensure the TensorWrapper remains in scope until the LabeledView goes out
 *  of scope.
 *
 *  In practice lifetime management is straightforward because LabeledView
 *  objects are typically unnamed temporaries. For example:
 *  ```
 *  TensorWrapper a,b,c;
 *  c("i,j") = a("i,k") * b("k,j");
 *  ```
 *  Here the three calls to TensorWrapper::operator(std::string) create
 *  three unnamed LabeledView instances, all of which are out of scope after
 *  the multiplication.
 *
 *  The only thing slightly tricky about this class is getting the const
 *  correctness right. In most expression layers this is done by having const
 *  show up somewhere in the type argument. We avoid this by having the class
 *  hold either a read/write reference or a read-only reference to the tensor.
 *  For the most part everything just works. The only hiccup comes from when you
 *  a non-const LabeledView instance, but it was initialized with a const
 *  reference (say from a const TensorWrapper object). In this case calling
 *  `tensor()` will throw as it would violate const-correctness by returing a
 *  read/write reference. Instead the user needs to call `tensor() const` to get
 *  back the read-only reference. In practice end users of TensorWrapper usually
 *  interact with this class implicitly, so this is only a complication seen by
 *  developers of TensorWrapper (it only really affects the Expression class).
 */
template<typename FieldType>
class LabeledView {
public:
    /// Type of the tensor this is a view of
    using tensor_type = TensorWrapper<FieldType>;

    /// Type of a read/write reference to the tensor this is a view of
    using tensor_reference = tensor_type&;

    /// Type of a read-only reference to the tensor this is a view of
    using const_tensor_reference = const tensor_type&;

    /// Type used to label the modes of the tensor
    using label_type = std::string;

    /// Type of a read-only reference to the tensor's labels
    using const_label_reference = const label_type&;

    LabeledView(label_type labels, tensor_reference tensor) noexcept;
    LabeledView(label_type labels, const_tensor_reference tensor) noexcept;

    /** @brief Creates a new LabeledView which is a copy of @p other.
     *
     *  This ctor will create a new LabeledView which is a deep copy of @p
     *  other. However, since views of a tensor have alias semantics, the deep-
     *  copy of the view also has alias semantics of the same tensor. The labels
     *  are actually deep copied.
     *
     *  @param[in] other The LabeledView we are deep copying.
     *
     *  @throw std::bad_alloc if there is a problem deep copying the labels.
     *                        Strong throw guarantee. (Copying the tensor
     *                        reference is no-throw guarantee.)
     */
    LabeledView(const LabeledView& other)     = default;
    LabeledView(LabeledView&& other) noexcept = default;
    ~LabeledView() noexcept                   = default;

    /** @brief Wraps this LabeledView in an Expression class.
     *
     *  The expression layer describes how pieces of tensor equations are
     *  combined. Every piece of the expression layer must be wrapped in an
     *  Expression instance. This method wraps the process of wrapping the
     *  current LabeledView instance in an Expression instance.
     */
    Expression<FieldType> expression() const;
    tensor_reference tensor();
    const_tensor_reference tensor() const;
    const_label_reference labels() const { return m_labels_; }

    /** @brief Overwrites the contents of this LabeledView with that of @p rhs
     *
     *  @warning In general this is NOT just copy assignment.
     */
    LabeledView& operator=(const LabeledView& rhs);
    LabeledView& operator=(const Expression& rhs);
    // Expression<FieldType> operator+(const LabeledView& rhs) const;
    // Expression<FieldType> operator*(double rhs) const;

private:
    using internal_reference       = std::reference_wrapper<tensor_type>;
    using internal_const_reference = std::reference_wrapper<const tensor_type>;

    label_type m_labels_;

    std::optional<internal_reference> m_tensor_;
    std::optional<internal_const_reference> m_ctensor_;
};

template<typename FieldType>
Expression<FieldType> operator*(double lhs, const LabeledView<FieldType>& rhs) {
    return rhs * lhs;
}

extern template class LabeledView<field::Scalar>;
extern template class LabeledView<field::Tensor>;

} // namespace expressions
} // namespace tensorwrapper::tensor
