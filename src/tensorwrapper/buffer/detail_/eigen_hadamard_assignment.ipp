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

#pragma once
#include "eigen_tensor.hpp"

namespace tensorwrapper::buffer::detail_ {

template<typename FloatType, unsigned int Rank>
void EigenTensor<FloatType, Rank>::hadamard_assignment_(
  label_type this_labels, label_type lhs_labels, label_type rhs_labels,
  const_pimpl_reference lhs, const_pimpl_reference rhs) {
    using my_type = EigenTensor<FloatType, Rank>;

    // Downcast LHS and RHS
    const auto* lhs_down  = dynamic_cast<const my_type*>(&lhs);
    const auto& lhs_eigen = lhs_down->value();
    const auto* rhs_down  = dynamic_cast<const my_type*>(&rhs);
    const auto& rhs_eigen = rhs_down->value();

    // Whose indices match whose?
    bool this_matches_lhs = (this_labels == lhs_labels);
    bool this_matches_rhs = (this_labels == rhs_labels);
    bool lhs_matches_rhs  = (lhs_labels == rhs_labels);

    // The three possible permutations we may need to apply
    auto get_permutation = [](auto&& lhs_, auto&& rhs_) {
        auto l_to_r = lhs_.permutation(rhs_);
        return std::vector<int>(l_to_r.begin(), l_to_r.end());
    };
    auto r_to_l    = get_permutation(rhs_labels, lhs_labels);
    auto l_to_r    = get_permutation(lhs_labels, rhs_labels);
    auto this_to_r = get_permutation(this_labels, rhs_labels);

    if(this_matches_lhs && this_matches_rhs) { // No permutations
        m_tensor_ = lhs_eigen * rhs_eigen;
    } else if(this_matches_lhs) { // RHS needs permuted
        m_tensor_ = lhs_eigen * rhs_eigen.shuffle(r_to_l);
    } else if(this_matches_rhs) { // LHS needs permuted
        m_tensor_ = lhs_eigen.shuffle(l_to_r) * rhs_eigen;
    } else if(lhs_matches_rhs) { // This needs permuted
        m_tensor_ = (lhs_eigen * rhs_eigen).shuffle(this_to_r);
    } else { // Everything needs permuted
        m_tensor_ = (lhs_eigen.shuffle(l_to_r) * rhs_eigen).shuffle(this_to_r);
    }
}
} // namespace tensorwrapper::buffer::detail_