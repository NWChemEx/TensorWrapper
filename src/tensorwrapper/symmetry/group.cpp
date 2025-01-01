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
#include <tensorwrapper/symmetry/group.hpp>

namespace tensorwrapper::symmetry {

using dsl_reference = typename Group::dsl_reference;

dsl_reference Group::addition_assignment_(label_type this_labels,
                                          const_labeled_reference rhs) {
    dsl::DummyIndices llabels(this_labels);
    dsl::DummyIndices rlabels(rhs.labels());

    // Make sure labels are a permutation of one another.
    auto p = rlabels.permutation(llabels);

    if(size() || rhs.object().size())
        throw std::runtime_error("Not sure how to propagate groups yet");

    return *this;
}

dsl_reference Group::permute_assignment_(label_type this_labels,
                                         const_labeled_reference rhs) {
    dsl::DummyIndices llabels(this_labels);
    dsl::DummyIndices rlabels(rhs.labels());

    // Make sure labels are a permutation of one another.
    auto p = rlabels.permutation(llabels);

    if(size() || rhs.object().size())
        throw std::runtime_error("Not sure how to propagate groups yet");

    return *this;
}

} // namespace tensorwrapper::symmetry