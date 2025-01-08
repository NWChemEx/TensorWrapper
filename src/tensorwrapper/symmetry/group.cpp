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
    return permute_assignment_(this_labels, lhs);
}

dsl_reference Group::permute_assignment_(label_type this_labels,
                                         const_labeled_reference rhs) {
    if(rhs.object().size() != 0)
        throw std::runtime_error("Support for non-trivial symmetry NYI!");

    return *this = rhs.object();
}

} // namespace tensorwrapper::symmetry