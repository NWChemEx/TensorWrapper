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

#include <tensorwrapper/symmetry/group.hpp>

namespace tensorwrapper::symmetry {
namespace {

template<typename LHSType, typename RHSType>
void assert_non_trivial(const LHSType& lhs, const RHSType& rhs) {
    if(lhs.object().size() != 0 || rhs.object().size() != 0)
        throw std::runtime_error("Support for non-trivial symmetry NYI!");
}

} // namespace

using dsl_reference = typename Group::dsl_reference;

dsl_reference Group::addition_assignment_(label_type this_labels,
                                          const_labeled_reference lhs,
                                          const_labeled_reference rhs) {
    assert_non_trivial(lhs, rhs);
    return permute_assignment_(this_labels, lhs);
}

dsl_reference Group::subtraction_assignment_(label_type this_labels,
                                             const_labeled_reference lhs,
                                             const_labeled_reference rhs) {
    assert_non_trivial(lhs, rhs);
    return permute_assignment_(this_labels, lhs);
}

dsl_reference Group::multiplication_assignment_(label_type this_labels,
                                                const_labeled_reference lhs,
                                                const_labeled_reference rhs) {
    assert_non_trivial(lhs, rhs);
    return *this = Group(this_labels.size());
}

dsl_reference Group::permute_assignment_(label_type this_labels,
                                         const_labeled_reference rhs) {
    if(rhs.object().size() != 0)
        throw std::runtime_error("Support for non-trivial symmetry NYI!");

    return *this = rhs.object();
}

} // namespace tensorwrapper::symmetry
