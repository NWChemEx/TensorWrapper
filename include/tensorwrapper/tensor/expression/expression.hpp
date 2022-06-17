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

    Expression operator+(Expression rhs) const;
    Expression operator*(Expression rhs) const;

    label_types output_labels(const label_type& lhs) const;
    tensor_type eval(const labeled_tensor& lhs) const;
    tot_type eval(const labled_tot& lhs) const;

private:
    explicit Expression(pimpl_pointer p);

    pimpl_pointer m_pimpl_;
};

// template<typename FieldType>
// Expression operator+(const LabeledTensor<FieldType>& lhs,
//                      const LabeledTensor<FieldType>& rhs) {
//     return Expression(lhs) + Expression(rhs);
// }

// template<typename LHSFieldType, typename RHSFieldType>
// Expression operator*(const LabeledTensor<LHSFieldType>& lhs,
//                      const LabeledTensor<RHSFieldType>& rhs) {
//     return Expression(lhs) * Expression(rhs);
// }

} // namespace tensorwrapper::tensor::expression
