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
                                           const_labeled_reference rhs) {
    // Computes the permutation necessary to permute rhs into *this
    auto ptemp = rhs.lhs().permute(rhs.rhs(), this_labels);

    // After permuting the shapes need to be equal for addition
    if(!ptemp->are_equal(*this))
        throw std::runtime_error("Shape " + ptemp->to_string() +
                                 " is not compatible for addition with " +
                                 this->to_string());

    // Ultimately addition assignment doesn't change the shape of *this so...
    return *this;
}

dsl_reference Smooth::permute_assignment_(label_type this_labels,
                                          const_labeled_reference rhs) {
    dsl::DummyIndices out_labels(this_labels);
    dsl::DummyIndices in_labels(rhs.rhs());

    if(in_labels.size() != rhs.lhs().rank())
        throw std::runtime_error("Incorrect number of indices");

    // This checks that out_labels is consistent with in_labels
    auto p          = in_labels.permutation(out_labels);
    auto smooth_rhs = rhs.lhs().as_smooth();

    extents_type temp(p.size());
    for(typename extents_type::size_type i = 0; i < p.size(); ++i)
        temp[p[i]] = smooth_rhs.extent(i);
    m_extents_.swap(temp);

    return *this;
}

} // namespace tensorwrapper::shape