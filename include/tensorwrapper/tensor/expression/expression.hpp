#pragma once
#include <tensorwrapper/tensor/expression/labeled_tensor.hpp>

namespace tensorwrapper::tensor::expression {
namespace detail_ {
class ExpressionPIMPL;
}

class Expression {
public:
    using labeled_tensor = LabeledTensor<Field::Scalar>;
    using tensor_type    = typename labeled_tensor::tensor_type;
    using labeled_tot    = LabeledTensor<Field::Tensor>;
    using tot_type       = typename labeled_tot::tensor_type;

    using label_type = typename labeled_tensor::label_type;

    using pimpl_type    = detail_::ExpressionPIMPL;
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    Expression(labeled_tensor t);
    Expression(labeled_tot tot);

    labeled_tensor eval(const labeled_tensor& result) const;
    labeled_tot eval(const labled_tot& result) const;

private:
    explicit Expression(pimpl_pointer p);

    pimpl_pointer m_pimpl_;
};

} // namespace tensorwrapper::tensor::expression
