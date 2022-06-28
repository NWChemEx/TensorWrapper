#pragma once
#include <stdexcept>
#include <tensorwrapper/tensor/expressions/expression.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

namespace tensorwrapper::tensor::expressions::detail_ {

template<typename FieldType>
class ExpressionPIMPL {
public:
    using expression_type = Expression<FieldType>;
    using labeled_tensor  = typename expression_type::labeled_tensor;
    using pimpl_pointer   = typename expression_type::pimpl_pointer;

    ExpressionPIMPL() noexcept          = default;
    virtual ~ExpressionPIMPL() noexcept = default;

    pimpl_pointer clone() const { return clone_(); }

    labeled_tensor& eval(labeled_tensor& lhs) const { return eval_(lhs); }

protected:
    ExpressionPIMPL(const ExpressionPIMPL& other) = default;
    ExpressionPIMPL(ExpressionPIMPL&& other)      = default;

    virtual pimpl_pointer clone_() const                     = 0;
    virtual labeled_tensor& eval_(labeled_tensor& lhs) const = 0;

private:
    ExpressionPIMPL& operator=(const ExpressionPIMPL& other) = delete;
};

} // namespace tensorwrapper::tensor::expressions::detail_
