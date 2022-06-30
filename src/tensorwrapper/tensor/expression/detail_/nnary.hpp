#pragma once

#include "pimpl.hpp"

namespace tensorwrapper::tensor::expression::detail_ {

template<typename FieldType, typename DerivedType, typename... Args>
class NNary : public ExpressionPIMPL<FieldType> {
private:
    using my_type   = NNary<FieldType, DerivedType, Args...>;
    using base_type = ExpressionPIMPL<FieldType>;

public:
    using typename base_type::pimpl_pointer;

    static constexpr std::size_t n_args = sizeof...(Args);

    template<typename... ArgsIn>
    explicit NNary(ArgsIn&&... args) : m_args_(std::forward<ArgsIn>(args)...) {}

    template<std::size_t I>
    const auto& arg() const {
        return std::get<I>(m_args_);
    }

protected:
    pimpl_pointer clone_() const override;
    bool are_equal_(const base_type& rhs) const noexcept override;

private:
    DerivedType& downcast_();
    const DerivedType& downcast_() const;
    std::tuple<Args...> m_args_;
};

#define TPARAMS \
    template<typename FieldType, typename DerivedType, typename... Args>
#define NNARY NNary<FieldType, DerivedType, Args...>

TPARAMS
typename NNARY::pimpl_pointer NNARY::clone_() const {
    return std::make_unique<DerivedType>(downcast_());
}

TPARAMS
bool NNARY::are_equal_(const base_type& rhs) const noexcept {
    const auto* prhs = dynamic_cast<const my_type*>(&rhs);
    if(prhs == nullptr) return false;
    return m_args_ == prhs->m_args_;
}

TPARAMS
DerivedType& NNARY::downcast_() { return *static_cast<DerivedType*>(this); }

TPARAMS
const DerivedType& NNARY::downcast_() const {
    return *static_cast<const DerivedType*>(this);
}

#undef NNARY
#undef TPARAMS

template<typename FieldType, typename DerivedType>
using LabeledBase = NNary<FieldType, DerivedType, LabeledView<FieldType>>;

template<typename FieldType, typename DerivedType>
using Binary =
  NNary<FieldType, DerivedType, Expression<FieldType>, Expression<FieldType>>;

template<typename FieldType, typename DerivedType, typename ScalarType>
using ScaleBase =
  NNary<FieldType, DerivedType, Expression<FieldType>, ScalarType>;

} // namespace tensorwrapper::tensor::expression::detail_
