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
void EigenTensor<FloatType, Rank>::permute_assignment_(
  label_type this_labels, label_type rhs_labels, const_pimpl_reference rhs) {
    using my_type        = EigenTensor<FloatType, Rank>;
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    if(this_labels != rhs_labels) { // We need to permute rhs before assignment
        // Eigen adopts the opposite definition of permutation from us.
        auto r_to_l = this_labels.permutation(rhs_labels);
        // Eigen wants int objects
        std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
        m_tensor_ = rhs_down->value().shuffle(r_to_l2);
    } else {
        m_tensor_ = rhs_down->value();
    }
}

} // namespace tensorwrapper::buffer::detail_