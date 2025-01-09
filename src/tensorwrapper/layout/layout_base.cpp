#include <tensorwrapper/layout/layout_base.hpp>

namespace tensorwrapper::layout {

using dsl_reference = typename LayoutBase::dsl_reference;

dsl_reference LayoutBase::addition_assignment_(label_type this_labels,
                                               const_labeled_reference lhs,
                                               const_labeled_reference rhs) {
    const auto& lobject = lhs.object();
    const auto& llabels = lhs.labels();
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    m_shape_->addition_assignment(this_labels, lobject.shape()(llabels),
                                  robject.shape()(rlabels));
    m_sparsity_->addition_assignment(this_labels, lobject.sparsity()(llabels),
                                     robject.sparsity()(rlabels));
    m_symmetry_->addition_assignment(this_labels, lobject.symmetry()(llabels),
                                     robject.symmetry()(rlabels));
    return *this;
}

dsl_reference LayoutBase::subtraction_assignment_(label_type this_labels,
                                                  const_labeled_reference lhs,
                                                  const_labeled_reference rhs) {
    const auto& lobject = lhs.object();
    const auto& llabels = lhs.labels();
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    m_shape_->subtraction_assignment(this_labels, lobject.shape()(llabels),
                                     robject.shape()(rlabels));
    m_sparsity_->subtraction_assignment(
      this_labels, lobject.sparsity()(llabels), robject.sparsity()(rlabels));
    m_symmetry_->subtraction_assignment(
      this_labels, lobject.symmetry()(llabels), robject.symmetry()(rlabels));
    return *this;
}

dsl_reference LayoutBase::multiplication_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    const auto& lobject = lhs.object();
    const auto& llabels = lhs.labels();
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    m_shape_->multiplication_assignment(this_labels, lobject.shape()(llabels),
                                        robject.shape()(rlabels));
    m_sparsity_->multiplication_assignment(
      this_labels, lobject.sparsity()(llabels), robject.sparsity()(rlabels));
    m_symmetry_->multiplication_assignment(
      this_labels, lobject.symmetry()(llabels), robject.symmetry()(rlabels));
    return *this;
}

dsl_reference LayoutBase::permute_assignment_(label_type this_labels,
                                              const_labeled_reference rhs) {
    const auto& robject = rhs.object();
    const auto& rlabels = rhs.labels();

    m_shape_->permute_assignment(this_labels, robject.shape()(rlabels));
    m_sparsity_->permute_assignment(this_labels, robject.sparsity()(rlabels));
    m_symmetry_->permute_assignment(this_labels, robject.symmetry()(rlabels));
    return *this;
}

} // namespace tensorwrapper::layout