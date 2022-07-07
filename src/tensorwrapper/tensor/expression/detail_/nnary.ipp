// This file meant only for inclusion in nnary.hpp

namespace tensorwrapper::tensor::expression::detail_ {

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

} // namespace tensorwrapper::tensor::expression::detail_
