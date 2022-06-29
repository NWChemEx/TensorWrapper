#pragma once
#include <memory>
#include <tensorwrapper/tensor/fields.hpp>

namespace tensorwrapper::tensor::expression {
namespace detail_ {
template<typename FieldType>
class ExpressionPIMPL;
}
template<typename FieldType>
class LabeledView;

template<typename FieldType>
class Expression {
public:
    using labeled_tensor = LabeledView<FieldType>;

    using pimpl_type    = detail_::ExpressionPIMPL<FieldType>;
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    explicit Expression(pimpl_pointer p = nullptr) noexcept;

    Expression(const Expression& other);

    Expression(Expression&& other) noexcept;

    /// Default no-throw dtor
    ~Expression() noexcept;

    Expression operator+(const Expression& rhs) const;
    Expression operator*(double rhs) const;
    Expression operator*(const Expression& rhs) const;

    labeled_tensor& eval(labeled_tensor& result) const;

    bool operator==(const Expression& rhs) const noexcept;
    bool operator!=(const Expression& rhs) const noexcept;

private:
    using const_pimpl_reference = const pimpl_type&;

    const_pimpl_reference pimpl_() const;

    pimpl_pointer m_pimpl_;
};

extern template class Expression<field::Scalar>;
extern template class Expression<field::Tensor>;

} // namespace tensorwrapper::tensor::expression
