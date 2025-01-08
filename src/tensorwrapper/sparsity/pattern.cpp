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

#include <tensorwrapper/sparsity/pattern.hpp>

namespace tensorwrapper::sparsity {

using dsl_reference = typename Pattern::dsl_reference;

dsl_reference Pattern::addition_assignment_(label_type this_labels,
                                            const_labeled_reference lhs,
                                            const_labeled_reference rhs) {
    return permute_assignment_(this_labels, lhs);
}

dsl_reference Pattern::subtraction_assignment_(label_type this_labels,
                                               const_labeled_reference lhs,
                                               const_labeled_reference rhs) {
    return permute_assignment_(this_labels, lhs);
}

dsl_reference Pattern::multiplication_assignment_(label_type this_labels,
                                                  const_labeled_reference lhs,
                                                  const_labeled_reference rhs) {
    return permute_assignment_(this_labels, lhs);
}

dsl_reference Pattern::permute_assignment_(label_type this_labels,
                                           const_labeled_reference rhs) {
    return *this = rhs.object();
}

} // namespace tensorwrapper::sparsity