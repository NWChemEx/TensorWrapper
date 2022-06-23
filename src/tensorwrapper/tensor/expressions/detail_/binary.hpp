#pragma once
#include "pimpl.hpp"

namespace tensorwrapper::tensor::expressions::detail_ {

/** @brief Code factorization for operations involving two expressions
 *
 *  Adding an operation to the expression layer involves a lot of boiler plate.
 *  The Binary class takes care of most of the boiler plate for operations
 *  that involve two expressions. Operations that occur pairwise should:
 *
 *  - inherit from this class
 *  - bring the Binary class's ctors into scope
 *  - define a public member `T& eval_common(T& )const` which actually does the
 *    operation (or override eval_ and implement the operation themselves)
 *
 *  @tparam DerivedType The class being implemented.
 */
template<typename DerivedType, typename LHSType, typename RHSType>
class Binary : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using expression_type = typename base_type::expression_type;
    using labeled_tensor  = typename base_type::labeled_tensor;
    using labeled_tot     = typename base_type::labeled_tot;
    using pimpl_pointer   = typename base_type::pimpl_pointer;

    Binary(LHSType lhs, RHSType rhs);
    Binary(const Binary& other) = default;

protected:
    pimpl_pointer clone_() const override;
    // labeled_tensor& eval_(labeled_tensor& result) const override;
    // labeled_tot& eval_(labeled_tot& result) const override;

    LHSType m_lhs_;
    RHSType m_rhs_;

private:
    const DerivedType& downcast_() const;
};

#define TPARAMS \
    template<typename DerivedType, typename LHSType, typename RHSType>
#define BINARY Binary<DerivedType, LHSType, RHSType>

TPARAMS
BINARY::Binary(LHSType lhs, RHSType rhs) :
  m_lhs_(std::move(lhs)), m_rhs_(std::move(rhs)) {}

TPARAMS
typename BINARY::pimpl_pointer BINARY::clone_() const {
    const auto* dcast = static_cast<const DerivedType*>(this);
    return std::make_unique<DerivedType>(*dcast);
}

// TPARAMS
// typename BINARY::labeled_tensor& BINARY::eval_(labeled_tensor& result) const
// {
//     return downcast_().eval_common(result);
// }

// TPARAMS
// inline typename BINARY::labeled_tot& BINARY::eval_(labeled_tot& result) const
// {
//     return downcast_().eval_common(result);
// }

TPARAMS
const DerivedType& BINARY::downcast_() const {
    return *(static_cast<const DerivedType*>(this));
}

#undef BINARY
#undef TPARAMS

} // namespace tensorwrapper::tensor::expressions::detail_
