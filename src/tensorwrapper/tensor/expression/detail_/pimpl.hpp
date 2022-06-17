#pragma once
#include <tensorwrapper/tensor/expression/expression.hpp>

namespace tensorwrapper::tensor::expression::detail_ {

class ExpressionPIMPL {
public:
    using expression_type = Expression;
    using labeled_tensor  = typename expression_type::labeled_tensor;
    using labeled_tot     = typename expression_type::labeled_tot;
    using tensor_type     = typename expression_type::tensor_type;
    using tot_type        = typename expression_type::tot_type;

    virtual ~ExpressionPIMPL() noexcept = default;

    label_type output_labels(const label_type& lhs) const {
        return output_labels_(lhs);
    }
    tensor_type eval(const labeled_tensor& lhs) const { return eval_(lhs); }
    tot_type eval(const labeled_tot& lhs) const { return eval_(lhs); }

protected:
    virtual label_type output_labels_(const label_type& lhs) const = 0;
    virtual tensor_type eval_(const labeled_tensor& lhs) const     = 0;
    virtual tot_type eval_(const labeled_tot& lhs) const           = 0;
};

} // namespace tensorwrapper::tensor::expression::detail_
