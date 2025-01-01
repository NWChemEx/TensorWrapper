/*
 * Copyright 2024 NWChemEx-Project
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

#include <tensorwrapper/dsl/dummy_indices.hpp>
#include <tensorwrapper/layout/layout_base.hpp>

namespace tensorwrapper::layout {

using dsl_reference = typename LayoutBase::dsl_reference;

namespace {

class CallAdditionAssignment {
public:
    template<typename LHSType, typename LHSLabels, typename RHSType>
    decltype(auto) run(LHSType&& lhs, LHSLabels&& this_labels,
                       RHSType&& rhs) const {
        return lhs.addition_assignment(std::forward<LHSLabels>(this_labels),
                                       std::forward<RHSType>(rhs));
    }
};

class CallPermuteAssignment {
public:
    template<typename LHSType, typename LHSLabels, typename RHSType>
    decltype(auto) run(LHSType&& lhs, LHSLabels&& this_labels,
                       RHSType&& rhs) const {
        return lhs.permute_assignment(std::forward<LHSLabels>(this_labels),
                                      std::forward<RHSType>(rhs));
    }
};

template<typename FunctorType, typename LHSType, typename LHSLabels,
         typename RHSType>
void assignment_guts(LHSType&& lhs, LHSLabels&& this_labels, RHSType&& rhs) {
    FunctorType f;

    const auto& rhs_shape = rhs.object().shape();
    f.run(lhs.shape(), this_labels, rhs_shape(rhs.labels()));

    const auto& rhs_symmetry = rhs.object().symmetry();
    f.run(lhs.symmetry(), this_labels, rhs_symmetry(rhs.labels()));

    const auto& rhs_sparsity = rhs.object().sparsity();
    f.run(lhs.sparsity(), this_labels, rhs_sparsity(rhs.labels()));
}

} // namespace

dsl_reference LayoutBase::addition_assignment_(label_type this_labels,
                                               const_labeled_reference rhs) {
    assignment_guts<CallAdditionAssignment>(*this, this_labels, rhs);
    return *this;
}

dsl_reference LayoutBase::permute_assignment_(label_type this_labels,
                                              const_labeled_reference rhs) {
    assignment_guts<CallPermuteAssignment>(*this, this_labels, rhs);
    return *this;
}

} // namespace tensorwrapper::layout