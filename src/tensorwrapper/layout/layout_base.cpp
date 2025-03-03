/*
 * Copyright 2025 NWChemEx-Project
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

#include <tensorwrapper/layout/layout_base.hpp>

namespace tensorwrapper::layout {

using dsl_reference = typename LayoutBase::dsl_reference;

template<typename FxnType>
dsl_reference LayoutBase::binary_common_(FxnType&& fxn, label_type this_labels,
                                         const_labeled_reference lhs,
                                         const_labeled_reference rhs) {
    const auto& lobject = lhs.object();
    const auto& llabels = lhs.labels();
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    const auto& lshape = lobject.shape();
    const auto& rshape = robject.shape();

    if(!m_shape_) m_shape_ = lshape.clone();
    fxn(*m_shape_, this_labels, lshape(llabels), rshape(rlabels));

    const auto& lsymm = lobject.symmetry();
    const auto& rsymm = robject.symmetry();
    if(!m_symmetry_) m_symmetry_ = lsymm.clone();
    fxn(*m_symmetry_, this_labels, lsymm(llabels), rsymm(rlabels));

    const auto& lsparsity = lobject.sparsity();
    const auto& rsparsity = robject.sparsity();
    if(!m_sparsity_) m_sparsity_ = lsparsity.clone();
    fxn(*m_sparsity_, this_labels, lsparsity(llabels), rsparsity(rlabels));

    return *this;
}

dsl_reference LayoutBase::addition_assignment_(label_type this_labels,
                                               const_labeled_reference lhs,
                                               const_labeled_reference rhs) {
    auto lambda = [](auto&& result, auto&& result_labels, auto&& labeled_lhs,
                     auto&& labeled_rhs) {
        result.addition_assignment(result_labels, labeled_lhs, labeled_rhs);
    };
    return binary_common_(lambda, this_labels, lhs, rhs);
}

dsl_reference LayoutBase::subtraction_assignment_(label_type this_labels,
                                                  const_labeled_reference lhs,
                                                  const_labeled_reference rhs) {
    auto lambda = [](auto&& result, auto&& result_labels, auto&& labeled_lhs,
                     auto&& labeled_rhs) {
        result.subtraction_assignment(result_labels, labeled_lhs, labeled_rhs);
    };
    return binary_common_(lambda, this_labels, lhs, rhs);
}

dsl_reference LayoutBase::multiplication_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    auto lambda = [](auto&& result, auto&& result_labels, auto&& labeled_lhs,
                     auto&& labeled_rhs) {
        result.multiplication_assignment(result_labels, labeled_lhs,
                                         labeled_rhs);
    };
    return binary_common_(lambda, this_labels, lhs, rhs);
}

dsl_reference LayoutBase::permute_assignment_(label_type this_labels,
                                              const_labeled_reference rhs) {
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    m_shape_->permute_assignment(this_labels, robject.shape()(rlabels));
    m_sparsity_->permute_assignment(this_labels, robject.sparsity()(rlabels));
    m_symmetry_->permute_assignment(this_labels, robject.symmetry()(rlabels));
    return *this;
}

} // namespace tensorwrapper::layout