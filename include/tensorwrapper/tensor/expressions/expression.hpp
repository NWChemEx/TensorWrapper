#pragma once
#include <memory>
#include <tensorwrapper/tensor/fields.hpp>

namespace tensorwrapper::tensor::expressions {
namespace detail_ {
class ExpressionPIMPL;
}
template<typename FieldType>
class LabeledTensor;

class Expression {
public:
    using labeled_tensor = LabeledTensor<field::Scalar>;
    using labeled_tot    = LabeledTensor<field::Tensor>;
    using pimpl_type     = detail_::ExpressionPIMPL;
    using pimpl_pointer  = std::unique_ptr<pimpl_type>;

    explicit Expression(pimpl_pointer p = nullptr) noexcept;

    Expression(const Expression& other);

    Expression(Expression&& other) noexcept;

    /// Default no-throw dtor
    ~Expression() noexcept;

    Expression operator+(const Expression& rhs) const;
    Expression operator*(double rhs) const;
    Expression operator*(const Expression& rhs) const;

    labeled_tensor& eval(labeled_tensor& result) const;
    labeled_tot& eval(labeled_tot& result) const;

private:
    using const_pimpl_reference = const pimpl_type&;

    const_pimpl_reference pimpl_() const;

    pimpl_pointer m_pimpl_;
};

} // namespace tensorwrapper::tensor::expressions
