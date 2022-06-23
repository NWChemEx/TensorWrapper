#pragma once

namespace tensorwrapper::tensor::expressions::detail_ {

template<typename DerivedType, typename ValueType>
class Unary : public ExpressionPIMPL {
private:
    using base_type = ExpressionPIMPL;

public:
    using pimpl_pointer  = typename base_type::pimpl_pointer;
    using labeled_tensor = typename base_type::labeled_tensor;
    using labeled_tot    = tyepname base_type::labeled_tot;

    explicit Unary(ValueType value) : m_value_(std::move(value)) {}
    Unary(const Unary& other) = default;

protected:
    pimpl_pointer clone_() const override;

    ValueType m_value_;

private:
    const DerivedType& downcast_() const;
};

#define TPARAMS template<typename DerivedType, typename ValueType>
#define UNARY Unary<DerivedType, ValueType>

TPARAMS
typename UNARY::pimpl_pointer UNARY::clone_() const {
    return std::make_unique<DerivedType>(downcast_());
}

TPARAMS
const DerivedType& UNARY::downcast_() const {
    return *(static_cast<const DerivedType*>(this));
}

#undef UNARY
#undef TPARAMS

} // namespace tensorwrapper::tensor::expressions::detail_
