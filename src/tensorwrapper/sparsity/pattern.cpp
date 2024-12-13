#include <tensorwrapper/dsl/dummy_indices.hpp>
#include <tensorwrapper/sparsity/pattern.hpp>

namespace tensorwrapper::sparsity {

using dsl_reference = typename Pattern::dsl_reference;

dsl_reference Pattern::addition_assignment_(label_type this_labels,
                                            const_labeled_reference rhs) {
    dsl::DummyIndices llabels(this_labels);
    dsl::DummyIndices rlabels(rhs.rhs());

    // Make sure labels are a permutation of one another.
    auto p = rlabels.permutation(llabels);

    return *this;
}

dsl_reference Pattern::permute_assignment_(label_type this_labels,
                                           const_labeled_reference rhs) {
    dsl::DummyIndices llabels(this_labels);
    dsl::DummyIndices rlabels(rhs.rhs());

    // Make sure labels are a permutation of one another.
    auto p = rlabels.permutation(llabels);

    return *this;
}

} // namespace tensorwrapper::sparsity