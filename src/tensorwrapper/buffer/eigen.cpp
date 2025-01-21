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
#include "eigen_contraction.hpp"
#include <sstream>
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/dsl/dummy_indices.hpp>

namespace tensorwrapper::buffer {

#define TPARAMS template<typename FloatType, unsigned short Rank>
#define EIGEN Eigen<FloatType, Rank>

using const_labeled_reference =
  typename Eigen<float, 0>::const_labeled_reference;
using dsl_reference = typename Eigen<float, 0>::dsl_reference;

TPARAMS
typename EIGEN::dsl_reference EIGEN::addition_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    BufferBase::addition_assignment_(this_labels, lhs, rhs);

    using allocator_type       = allocator::Eigen<FloatType, Rank>;
    const auto& lhs_downcasted = allocator_type::rebind(lhs.object());
    const auto& rhs_downcasted = allocator_type::rebind(rhs.object());
    const auto& lhs_eigen      = lhs_downcasted.value();
    const auto& rhs_eigen      = rhs_downcasted.value();

    const auto& lhs_labels = lhs.labels();
    const auto& rhs_labels = rhs.labels();

    bool this_matches_lhs = (this_labels == lhs_labels);
    bool this_matches_rhs = (this_labels == rhs_labels);
    bool lhs_matches_rhs  = (lhs_labels == rhs_labels);

    auto get_permutation = [](auto&& lhs_, auto&& rhs_) {
        auto l_to_r = lhs_.permutation(rhs_);
        return std::vector<int>(l_to_r.begin(), l_to_r.end());
    };

    auto r_to_l    = get_permutation(rhs_labels, lhs_labels);
    auto l_to_r    = get_permutation(lhs_labels, rhs_labels);
    auto this_to_r = get_permutation(this_labels, rhs_labels);

    if(this_matches_lhs && this_matches_rhs) { // No permutations
        m_tensor_ = lhs_eigen + rhs_eigen;
    } else if(this_matches_lhs) { // RHS needs permuted
        m_tensor_ = lhs_eigen + rhs_eigen.shuffle(r_to_l);
    } else if(this_matches_rhs) { // LHS needs permuted
        m_tensor_ = lhs_eigen.shuffle(l_to_r) + rhs_eigen;
    } else if(lhs_matches_rhs) { // This needs permuted
        m_tensor_ = (lhs_eigen + rhs_eigen).shuffle(this_to_r);
    } else { // Everything needs permuted
        m_tensor_ = (lhs_eigen.shuffle(l_to_r) + rhs_eigen).shuffle(this_to_r);
    }

    return *this;
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::subtraction_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    BufferBase::subtraction_assignment_(this_labels, lhs, rhs);

    using allocator_type       = allocator::Eigen<FloatType, Rank>;
    const auto& lhs_downcasted = allocator_type::rebind(lhs.object());
    const auto& rhs_downcasted = allocator_type::rebind(rhs.object());
    const auto& lhs_eigen      = lhs_downcasted.value();
    const auto& rhs_eigen      = rhs_downcasted.value();

    const auto& lhs_labels = lhs.labels();
    const auto& rhs_labels = rhs.labels();

    bool this_matches_lhs = (this_labels == lhs_labels);
    bool this_matches_rhs = (this_labels == rhs_labels);
    bool lhs_matches_rhs  = (lhs_labels == rhs_labels);

    auto get_permutation = [](auto&& lhs_, auto&& rhs_) {
        auto l_to_r = lhs_.permutation(rhs_);
        return std::vector<int>(l_to_r.begin(), l_to_r.end());
    };

    auto r_to_l    = get_permutation(rhs_labels, lhs_labels);
    auto l_to_r    = get_permutation(lhs_labels, rhs_labels);
    auto this_to_r = get_permutation(this_labels, rhs_labels);

    if(this_matches_lhs && this_matches_rhs) { // No permutations
        m_tensor_ = lhs_eigen - rhs_eigen;
    } else if(this_matches_lhs) { // RHS needs permuted
        m_tensor_ = lhs_eigen - rhs_eigen.shuffle(r_to_l);
    } else if(this_matches_rhs) { // LHS needs permuted
        m_tensor_ = lhs_eigen.shuffle(l_to_r) - rhs_eigen;
    } else if(lhs_matches_rhs) { // This needs permuted
        m_tensor_ = (lhs_eigen - rhs_eigen).shuffle(this_to_r);
    } else { // Everything needs permuted
        m_tensor_ = (lhs_eigen.shuffle(l_to_r) - rhs_eigen).shuffle(this_to_r);
    }

    return *this;
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::multiplication_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    BufferBase::multiplication_assignment_(this_labels, lhs, rhs);

    if(this_labels.is_hadamard_product(lhs.labels(), rhs.labels()))
        return hadamard_(this_labels, lhs, rhs);
    else if(this_labels.is_contraction(lhs.labels(), rhs.labels()))
        return contraction_(this_labels, lhs, rhs);
    else
        throw std::runtime_error("Mixed products NYI");
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::permute_assignment_(
  label_type this_labels, const_labeled_reference rhs) {
    BufferBase::permute_assignment_(this_labels, rhs);

    using allocator_type       = allocator::Eigen<FloatType, Rank>;
    const auto& rhs_downcasted = allocator_type::rebind(rhs.object());

    const auto& rlabels = rhs.labels();

    if(this_labels != rlabels) { // We need to permute rhs before assignment
        auto r_to_l = rhs.labels().permutation(this_labels);
        // Eigen wants int objects
        std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
        m_tensor_ = rhs_downcasted.value().shuffle(r_to_l2);
    } else {
        m_tensor_ = rhs_downcasted.value();
    }

    return *this;
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::scalar_multiplication_(
  label_type this_labels, double scalar, const_labeled_reference rhs) {
    BufferBase::permute_assignment_(this_labels, rhs);

    using allocator_type       = allocator::Eigen<FloatType, Rank>;
    const auto& rhs_downcasted = allocator_type::rebind(rhs.object());

    const auto& rlabels = rhs.labels();

    FloatType c(scalar);

    if(this_labels != rlabels) { // We need to permute rhs before assignment
        auto r_to_l = rhs.labels().permutation(this_labels);
        // Eigen wants int objects
        std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
        m_tensor_ = rhs_downcasted.value().shuffle(r_to_l2) * c;
    } else {
        m_tensor_ = rhs_downcasted.value() * c;
    }

    return *this;
}

TPARAMS
typename detail_::PolymorphicBase<BufferBase>::string_type EIGEN::to_string_()
  const {
    std::stringstream ss;
    ss << m_tensor_;
    return ss.str();
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::hadamard_(label_type this_labels,
                                               const_labeled_reference lhs,
                                               const_labeled_reference rhs) {
    using allocator_type       = allocator::Eigen<FloatType, Rank>;
    const auto& lhs_downcasted = allocator_type::rebind(lhs.object());
    const auto& rhs_downcasted = allocator_type::rebind(rhs.object());
    const auto& lhs_eigen      = lhs_downcasted.value();
    const auto& rhs_eigen      = rhs_downcasted.value();

    const auto& lhs_labels = lhs.labels();
    const auto& rhs_labels = rhs.labels();

    bool this_matches_lhs = (this_labels == lhs_labels);
    bool this_matches_rhs = (this_labels == rhs_labels);
    bool lhs_matches_rhs  = (lhs_labels == rhs_labels);

    auto get_permutation = [](auto&& lhs_, auto&& rhs_) {
        auto l_to_r = lhs_.permutation(rhs_);
        return std::vector<int>(l_to_r.begin(), l_to_r.end());
    };

    auto r_to_l    = get_permutation(rhs_labels, lhs_labels);
    auto l_to_r    = get_permutation(lhs_labels, rhs_labels);
    auto this_to_r = get_permutation(this_labels, rhs_labels);

    if(this_matches_lhs && this_matches_rhs) { // No permutations
        m_tensor_ = lhs_eigen * rhs_eigen;
    } else if(this_matches_lhs) { // RHS needs permuted
        m_tensor_ = lhs_eigen * rhs_eigen.shuffle(r_to_l);
    } else if(this_matches_rhs) { // LHS needs permuted
        m_tensor_ = lhs_eigen.shuffle(l_to_r) * rhs_eigen;
    } else if(lhs_matches_rhs) { // This needs permuted
        m_tensor_ = (lhs_eigen * rhs_eigen).shuffle(this_to_r);
    } else { // Everything needs permuted
        m_tensor_ = (lhs_eigen.shuffle(l_to_r) * rhs_eigen).shuffle(this_to_r);
    }

    return *this;
}

TPARAMS typename EIGEN::dsl_reference EIGEN::contraction_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    const auto& llabels = lhs.labels();
    const auto& lobject = lhs.object();
    const auto& rlabels = rhs.labels();
    const auto& robject = rhs.object();

    // N.b. is a pure contraction, so common indices are summed over
    auto common = llabels.intersection(rlabels);

    // -- This block converts string indices to mode offsets
    using rank_type = unsigned short;
    using pair_type = std::pair<rank_type, rank_type>;
    std::vector<pair_type> modes;
    auto rank = common.size();
    for(decltype(rank) i = 0; i < rank; ++i) {
        const auto& index_i = common.at(i);
        // N.b., pure contraction so there's no repeats within a tensor's label
        auto lindex = llabels.find(index_i)[0];
        auto rindex = rlabels.find(index_i)[0];
        modes.push_back(pair_type(lindex, rindex));
    }

    return eigen_contraction<FloatType>(*this, lobject, robject, modes);
}

#undef EIGEN
#undef TPARAMS

#define DEFINE_EIGEN_BUFFER(TYPE)        \
    template class Eigen<TYPE, 0>; \
    template class Eigen<TYPE, 1>; \
    template class Eigen<TYPE, 2>; \
    template class Eigen<TYPE, 3>; \
    template class Eigen<TYPE, 4>; \
    template class Eigen<TYPE, 5>; \
    template class Eigen<TYPE, 6>; \
    template class Eigen<TYPE, 7>; \
    template class Eigen<TYPE, 8>; \
    template class Eigen<TYPE, 9>; \
    template class Eigen<TYPE, 10>

DEFINE_EIGEN_BUFFER(float);
DEFINE_EIGEN_BUFFER(double);

#ifdef TENSORWRAPPER_HAS_SIGMA
DEFINE_EIGEN_BUFFER(sigma::UFloat);
DEFINE_EIGEN_BUFFER(sigma::UDouble);
#endif

#undef DEFINE_EIGEN_BUFFER

} // namespace tensorwrapper::buffer