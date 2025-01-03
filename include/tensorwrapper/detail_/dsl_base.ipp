/** @file dsl_base.ipp
 *
 *  Contains inline implementations for the DSLBase class. This file is meant
 *  only for inclusion by dsl_base.hpp.
 */

namespace tensorwrapper::detail_ {

#define TPARAMS template<typename DerivedType, typename StringType>
#define DSL_BASE DSLBase<DerivedType, StringType>

TPARAMS
template<typename LabelType>
typename DSL_BASE::dsl_reference DSL_BASE::addition_assignment(
  LabelType&& this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    assert_indices_match_rank_(lhs);
    assert_indices_match_rank_(rhs);
    assert_is_permutation_(lhs.labels(), rhs.labels());

    label_type result_labels(std::forward<LabelType>(this_labels));
    auto lr_labels = lhs.labels().concatenation(rhs.labels());
    assert_is_subset_(result_labels, lr_labels);

    return addition_assignment_(std::move(result_labels), lhs, rhs);
}

TPARAMS
template<typename LabelType>
typename DSL_BASE::dsl_reference DSL_BASE::subtraction_assignment(
  LabelType&& this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    assert_indices_match_rank_(lhs);
    assert_indices_match_rank_(rhs);
    assert_is_permutation_(lhs.labels(), rhs.labels());

    label_type result_labels(std::forward<LabelType>(this_labels));
    auto lr_labels = lhs.labels().concatenation(rhs.labels());
    assert_is_subset_(result_labels, lr_labels);

    return subtraction_assignment_(std::move(result_labels), lhs, rhs);
}

TPARAMS
template<typename LabelType>
typename DSL_BASE::dsl_reference DSL_BASE::multiplication_assignment(
  LabelType&& this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    assert_indices_match_rank_(lhs);
    assert_indices_match_rank_(rhs);

    label_type result_labels(std::forward<LabelType>(this_labels));
    auto lr_labels = lhs.labels().concatenation(rhs.labels());
    assert_is_subset_(result_labels, lr_labels);

    return multiplication_assignment_(std::move(result_labels), lhs, rhs);
}

TPARAMS
template<typename LabelType>
typename DSL_BASE::dsl_reference DSL_BASE::permute_assignment(
  LabelType&& this_labels, const_labeled_reference rhs) {
    assert_indices_match_rank_(rhs);

    label_type lhs_labels(std::forward<LabelType>(this_labels));
    assert_is_subset_(lhs_labels, rhs.labels());

    return permute_assignment_(std::move(lhs_labels), rhs);
}

#undef DSL_BASE
#undef TPARAMS

} // namespace tensorwrapper::detail_