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
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::shape {

using dsl_reference = typename Smooth::dsl_reference;

dsl_reference Smooth::addition_assignment_(label_type this_labels,
                                           const_labeled_reference lhs,
                                           const_labeled_reference rhs) {
    // Ultimately addition doesn't change the shape unless there's a trace
    // or permutation. permute_assignment_ will take care of both scenarios.

    // The base class ensured that lhs and rhs are related by a permutation.
    // So all we have to do is permute either lhs or rhs into the final shape
    return permute_assignment(this_labels, rhs);
}

dsl_reference Smooth::subtraction_assignment_(label_type this_labels,
                                              const_labeled_reference lhs,
                                              const_labeled_reference rhs) {
    // Ultimately subtraction doesn't change the shape unless there's a trace
    // or permutation. permute_assignment_ will take care of both scenarios.

    // The base class ensured that lhs and rhs are related by a permutation.
    // So all we have to do is permute either lhs or rhs into the final shape
    return permute_assignment(this_labels, rhs);
}

dsl_reference Smooth::multiplication_assignment_(label_type this_labels,
                                                 const_labeled_reference lhs,
                                                 const_labeled_reference rhs) {
    const auto& labels_lhs = lhs.labels();
    const auto& labels_rhs = rhs.labels();
    auto smooth_lhs        = lhs.object().as_smooth();
    auto smooth_rhs        = rhs.object().as_smooth();

    // For each label
    // we will be able to find it in either lhs or rhs and then set temp[i] to
    // the corresponding extent
    extents_type temp(this_labels.size());
    for(size_type i = 0; i < this_labels.size(); ++i) {
        const auto& label_i = this_labels.at(i);

        if(labels_lhs.count(label_i)) {
            temp[i] = smooth_lhs.extent(labels_lhs.find(label_i)[0]);
        } else {
            // Base verified this_labels is a subset of lhs + rhs, so must be
            // in rhs
            temp[i] = smooth_rhs.extent(labels_rhs.find(label_i)[0]);
        }
    }
    m_extents_.swap(temp);
    return *this;
}

dsl_reference Smooth::permute_assignment_(label_type this_labels,
                                          const_labeled_reference rhs) {
    if(this_labels.size() != rhs.labels().size())
        throw std::runtime_error("Trace NYI");

    auto p          = rhs.labels().permutation(this_labels);
    auto smooth_rhs = rhs.object().as_smooth();

    extents_type temp(p.size());
    for(typename extents_type::size_type i = 0; i < p.size(); ++i)
        temp[p[i]] = smooth_rhs.extent(i);
    m_extents_.swap(temp);

    return *this;
}

} // namespace tensorwrapper::shape
