#include <tensorwrapper/dsl/dummy_indices.hpp>
#include <tensorwrapper/symmetry/group.hpp>

namespace tensorwrapper::symmetry {

using dsl_reference = typename Group::dsl_reference;

dsl_reference Group::addition_assignment_(label_type this_labels,
                                          const_labeled_reference rhs) {
    dsl::DummyIndices llabels(this_labels);
    dsl::DummyIndices rlabels(rhs.rhs());

    // Make sure labels are a permutation of one another.
    auto p = rlabels.permutation(llabels);

    if(size() || rhs.lhs().size())
        throw std::runtime_error("Not sure how to propagate groups yet");

    return *this;
}

dsl_reference Group::permute_assignment_(label_type this_labels,
                                         const_labeled_reference rhs) {
    dsl::DummyIndices llabels(this_labels);
    dsl::DummyIndices rlabels(rhs.rhs());

    // Make sure labels are a permutation of one another.
    auto p = rlabels.permutation(llabels);

    if(size() || rhs.lhs().size())
        throw std::runtime_error("Not sure how to propagate groups yet");

    return *this;
}

} // namespace tensorwrapper::symmetry