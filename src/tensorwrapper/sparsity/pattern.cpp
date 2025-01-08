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