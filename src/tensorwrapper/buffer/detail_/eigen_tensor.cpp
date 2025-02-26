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

#include "../../backends/eigen.hpp"
#include "../contraction_planner.hpp"
#include "eigen_tensor.hpp"

namespace tensorwrapper::buffer::detail_ {

#define TPARAMS template<typename FloatType, unsigned int Rank>
#define EIGEN_TENSOR EigenTensor<FloatType, Rank>

TPARAMS
template<typename OperationType>
void EIGEN_TENSOR::element_wise_op_(OperationType op, label_type this_labels,
                                    label_type lhs_labels,
                                    label_type rhs_labels,
                                    const_pimpl_reference lhs,
                                    const_pimpl_reference rhs) {
    // Downcast LHS and RHS
    const auto* lhs_down  = dynamic_cast<const my_type*>(&lhs);
    const auto& lhs_eigen = lhs_down->value();
    const auto* rhs_down  = dynamic_cast<const my_type*>(&rhs);
    const auto& rhs_eigen = rhs_down->value();

    // Whose indices match whose?
    bool this_matches_lhs = (this_labels == lhs_labels);
    bool this_matches_rhs = (this_labels == rhs_labels);
    bool lhs_matches_rhs  = (lhs_labels == rhs_labels);

    // The three possible permutations we may need to apply
    auto get_permutation = [](auto&& lhs_, auto&& rhs_) {
        auto l_to_r = lhs_.permutation(rhs_);
        return std::vector<int>(l_to_r.begin(), l_to_r.end());
    };
    auto r_to_l    = get_permutation(rhs_labels, lhs_labels);
    auto l_to_r    = get_permutation(lhs_labels, rhs_labels);
    auto this_to_r = get_permutation(this_labels, rhs_labels);

    if(this_matches_lhs && this_matches_rhs) { // No permutations
        m_tensor_ = op(lhs_eigen, rhs_eigen);
    } else if(this_matches_lhs) { // RHS needs permuted
        m_tensor_ = op(lhs_eigen, rhs_eigen.shuffle(r_to_l));
    } else if(this_matches_rhs) { // LHS needs permuted
        m_tensor_ = op(lhs_eigen.shuffle(l_to_r), rhs_eigen);
    } else if(lhs_matches_rhs) { // This needs permuted
        m_tensor_ = op(lhs_eigen, rhs_eigen).shuffle(this_to_r);
    } else { // Everything needs permuted
        m_tensor_ = op(lhs_eigen.shuffle(l_to_r), rhs_eigen).shuffle(this_to_r);
    }
}

TPARAMS
void EIGEN_TENSOR::addition_assignment_(label_type this_labels,
                                        label_type lhs_labels,
                                        label_type rhs_labels,
                                        const_pimpl_reference lhs,
                                        const_pimpl_reference rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs + rhs; };
    element_wise_op_(lambda, std::move(this_labels), std::move(lhs_labels),
                     std::move(rhs_labels), lhs, rhs);
}

template<typename TensorType>
auto matrix_size(TensorType&& t, std::size_t row_ranks) {
    std::size_t nrows = 1;
    for(std::size_t i = 0; i < row_ranks; ++i) nrows *= t.extent(i);

    std::size_t ncols = 1;
    const auto rank   = t.rank();
    for(std::size_t i = row_ranks; i < rank; ++i) ncols *= t.extent(i);
    return std::make_pair(nrows, ncols);
}

TPARAMS
void EIGEN_TENSOR::contraction_assignment_(label_type olabels,
                                           label_type llabels,
                                           label_type rlabels,
                                           const_shape_reference result_shape,
                                           const_pimpl_reference lhs,
                                           const_pimpl_reference rhs) {
    ContractionPlanner plan(olabels, llabels, rlabels);

    auto lt = lhs.clone();
    auto rt = rhs.clone();
    lt->permute_assignment(plan.lhs_permutation(), llabels, lhs);
    rt->permute_assignment(plan.rhs_permutation(), rlabels, rhs);

    const auto [lrows, lcols] = matrix_size(*lt, plan.lhs_free().size());
    const auto [rrows, rcols] = matrix_size(*rt, plan.rhs_dummy().size());

    // Work out the types of the matrix amd a map
    constexpr auto e_dyn       = ::Eigen::Dynamic;
    constexpr auto e_row_major = ::Eigen::RowMajor;
    using matrix_t = ::Eigen::Matrix<FloatType, e_dyn, e_dyn, e_row_major>;
    using map_t    = ::Eigen::Map<matrix_t>;

    eigen::data_type<FloatType, 2> buffer(lrows, rcols);

    map_t lmatrix(lt->data(), lrows, lcols);
    map_t rmatrix(rt->data(), rrows, rcols);
    map_t omatrix(buffer.data(), lrows, rcols);
    omatrix = lmatrix * rmatrix;

    auto mlabels = plan.result_matrix_labels();
    auto oshape  = result_shape(olabels);

    // oshapes is the final shape, permute it to shape omatrix is currently in
    auto temp_shape = result_shape.clone();
    temp_shape->permute_assignment(mlabels, oshape);
    auto mshape = temp_shape->as_smooth();

    auto m_to_o = olabels.permutation(mlabels); // N.b. Eigen def is inverse us

    std::array<int, Rank> out_size;
    std::array<int, Rank> m_to_o_array;
    for(std::size_t i = 0; i < Rank; ++i) {
        out_size[i]     = mshape.extent(i);
        m_to_o_array[i] = m_to_o[i];
    }

    auto tensor = buffer.reshape(out_size);
    if constexpr(Rank > 0) {
        m_tensor_ = tensor.shuffle(m_to_o_array);
    } else {
        m_tensor_ = tensor;
    }
}

TPARAMS
void EIGEN_TENSOR::hadamard_assignment_(label_type this_labels,
                                        label_type lhs_labels,
                                        label_type rhs_labels,
                                        const_pimpl_reference lhs,
                                        const_pimpl_reference rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs * rhs; };
    element_wise_op_(lambda, std::move(this_labels), std::move(lhs_labels),
                     std::move(rhs_labels), lhs, rhs);
}

TPARAMS
void EIGEN_TENSOR::permute_assignment_(label_type this_labels,
                                       label_type rhs_labels,
                                       const_pimpl_reference rhs) {
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    if(this_labels != rhs_labels) { // We need to permute rhs before assignment
        // Eigen adopts the opposite definition of permutation from us.
        auto r_to_l = this_labels.permutation(rhs_labels);
        // Eigen wants int objects
        std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
        m_tensor_ = rhs_down->value().shuffle(r_to_l2);
    } else {
        m_tensor_ = rhs_down->value();
    }
}

TPARAMS
void EIGEN_TENSOR::scalar_multiplication_(label_type this_labels,
                                          label_type rhs_labels,
                                          FloatType scalar,
                                          const_pimpl_reference rhs) {
    const auto* rhs_downcasted = dynamic_cast<const my_type*>(&rhs);

    if(this_labels != rhs_labels) { // We need to permute rhs before assignment
        auto r_to_l = rhs_labels.permutation(this_labels);
        // Eigen wants int objects
        std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
        m_tensor_ = rhs_downcasted->value().shuffle(r_to_l2) * scalar;
    } else {
        m_tensor_ = rhs_downcasted->value() * scalar;
    }
}

TPARAMS
void EIGEN_TENSOR::subtraction_assignment_(label_type this_labels,
                                           label_type lhs_labels,
                                           label_type rhs_labels,
                                           const_pimpl_reference lhs,
                                           const_pimpl_reference rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs - rhs; };
    element_wise_op_(lambda, std::move(this_labels), std::move(lhs_labels),
                     std::move(rhs_labels), lhs, rhs);
}

#undef EIGEN_TENSOR
#undef TPARAMS

#define DEFINE_EIGEN_TENSOR(TYPE)        \
    template class EigenTensor<TYPE, 0>; \
    template class EigenTensor<TYPE, 1>; \
    template class EigenTensor<TYPE, 2>; \
    template class EigenTensor<TYPE, 3>; \
    template class EigenTensor<TYPE, 4>; \
    template class EigenTensor<TYPE, 5>; \
    template class EigenTensor<TYPE, 6>; \
    template class EigenTensor<TYPE, 7>; \
    template class EigenTensor<TYPE, 8>; \
    template class EigenTensor<TYPE, 9>; \
    template class EigenTensor<TYPE, 10>

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_EIGEN_TENSOR);

#undef DEFINE_EIGEN_TENSOR

} // namespace tensorwrapper::buffer::detail_