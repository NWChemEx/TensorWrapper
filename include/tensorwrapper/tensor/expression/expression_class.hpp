#pragma once
#include <memory>
#include <tensorwrapper/tensor/field/traits.hpp>

namespace tensorwrapper::tensor::expression {
namespace detail_ {

template<typename FieldType>
class ExpressionPIMPL;

}

template<typename FieldType>
class Expression {
private:
    using ft = field::FieldTraits<FieldType>;

public:
    using const_label_reference     = typename ft::const_label_reference;
    using tensor_type               = typename ft::tensor_type;
    using const_allocator_reference = typename ft::const_allocator_reference;
    using const_shape_reference     = typename ft::const_shape_reference;

    using pimpl_type    = detail_::ExpressionPIMPL<FieldType>;
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    explicit Expression(pimpl_pointer p = nullptr) noexcept;

    Expression(const Expression& other);

    Expression(Expression&& other) noexcept;

    /// Default no-throw dtor
    ~Expression() noexcept;

    Expression operator+(const Expression& rhs) const;
    Expression operator-(const Expression& rhs) const;
    Expression operator*(double rhs) const;
    Expression operator*(const Expression& rhs) const;

    tensor_type tensor(const_label_reference labels,
                       const_shape_reference shape,
                       const_allocator_reference alloc) const;

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
