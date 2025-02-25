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
void EigenTensor<FloatType, Rank>::scalar_multiplication_(
  label_type this_labels, label_type rhs_labels, FloatType scalar,
  const_pimpl_reference rhs) {
    using my_type              = EigenTensor<FloatType, Rank>;
    const auto* rhs_downcasted = dynamic_cast<const my_type*>(&rhs);

    if(this_labels != rhs_labels) { // We need to permute rhs before assignment
        auto r_to_l = rhs_labels.permutation(this_labels);
        // Eigen wants int objects
        std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
        m_tensor_ = rhs_downcasted->value().shuffle(r_to_l2) * scalar;
    } else {
        m_tensor_ = rhs_downcasted->value() * scalar;
    }
}
} // namespace tensorwrapper::buffer::detail_