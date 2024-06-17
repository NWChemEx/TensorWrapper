/*
 * Copyright 2022 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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
