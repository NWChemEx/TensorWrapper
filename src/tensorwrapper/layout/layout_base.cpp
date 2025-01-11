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

dsl_reference LayoutBase::addition_assignment_(label_type this_labels,
                                               const_labeled_reference lhs,
                                               const_labeled_reference rhs) {
    const auto& lobject = lhs.object();
    const auto& llabels = lhs.labels();
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    m_shape_->addition_assignment(this_labels, lobject.shape()(llabels),
                                  robject.shape()(rlabels));
    m_sparsity_->addition_assignment(this_labels, lobject.sparsity()(llabels),
                                     robject.sparsity()(rlabels));
    m_symmetry_->addition_assignment(this_labels, lobject.symmetry()(llabels),
                                     robject.symmetry()(rlabels));
    return *this;
}

dsl_reference LayoutBase::subtraction_assignment_(label_type this_labels,
                                                  const_labeled_reference lhs,
                                                  const_labeled_reference rhs) {
    const auto& lobject = lhs.object();
    const auto& llabels = lhs.labels();
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    m_shape_->subtraction_assignment(this_labels, lobject.shape()(llabels),
                                     robject.shape()(rlabels));
    m_sparsity_->subtraction_assignment(
      this_labels, lobject.sparsity()(llabels), robject.sparsity()(rlabels));
    m_symmetry_->subtraction_assignment(
      this_labels, lobject.symmetry()(llabels), robject.symmetry()(rlabels));
    return *this;
}

dsl_reference LayoutBase::multiplication_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    const auto& lobject = lhs.object();
    const auto& llabels = lhs.labels();
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    m_shape_->multiplication_assignment(this_labels, lobject.shape()(llabels),
                                        robject.shape()(rlabels));
    m_sparsity_->multiplication_assignment(
      this_labels, lobject.sparsity()(llabels), robject.sparsity()(rlabels));
    m_symmetry_->multiplication_assignment(
      this_labels, lobject.symmetry()(llabels), robject.symmetry()(rlabels));
    return *this;
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