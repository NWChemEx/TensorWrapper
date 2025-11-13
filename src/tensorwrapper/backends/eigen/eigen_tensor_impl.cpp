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

// #include "../contraction_planner.hpp"
#include "eigen_tensor_impl.hpp"
#include <iomanip>
#include <sstream>

#ifdef ENABLE_CUTENSOR
#include "eigen_tensor.cuh"
#endif

namespace tensorwrapper::backends::eigen {

std::vector<int> to_eigen_permutation(const symmetry::Permutation& perm) {
    std::vector<int> eigen_perm(perm.rank());
    std::iota(eigen_perm.begin(), eigen_perm.end(), 0);
    return perm.apply(std::move(eigen_perm));
}

#define TPARAMS template<typename FloatType, unsigned int Rank>
#define EIGEN_TENSOR EigenTensorImpl<FloatType, Rank>

TPARAMS
auto EIGEN_TENSOR::get_elem_(index_vector index) const -> const_reference {
    return unwrap_vector_(std::move(index), std::make_index_sequence<Rank>());
}

TPARAMS
void EIGEN_TENSOR::set_elem_(index_vector index, value_type new_value) {
    unwrap_vector_(std::move(index), std::make_index_sequence<Rank>()) =
      new_value;
}

TPARAMS
void EIGEN_TENSOR::fill_(value_type value) {
    std::fill(m_tensor_.data(), m_tensor_.data() + m_tensor_.size(), value);
}

TPARAMS
auto EIGEN_TENSOR::to_string_() const -> string_type {
    std::stringstream ss;
    add_to_stream_(ss);
    return ss.str();
}

TPARAMS
std::ostream& EIGEN_TENSOR::add_to_stream_(std::ostream& os) const {
    os << std::fixed << std::setprecision(16);
    return os << m_tensor_.format(Eigen::TensorIOFormat::Numpy());
}

TPARAMS
void EIGEN_TENSOR::addition_assignment_(const_permutation_reference lhs_permute,
                                        const_permutation_reference rhs_permute,
                                        const base_type& lhs,
                                        const base_type& rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs + rhs; };
    element_wise_op_(lambda, lhs_permute, rhs_permute, lhs, rhs);
}

TPARAMS
void EIGEN_TENSOR::subtraction_assignment_(
  const_permutation_reference lhs_permute,
  const_permutation_reference rhs_permute, const base_type& lhs,
  const base_type& rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs - rhs; };
    element_wise_op_(lambda, lhs_permute, rhs_permute, lhs, rhs);
}

TPARAMS
void EIGEN_TENSOR::hadamard_assignment_(const_permutation_reference lhs_permute,
                                        const_permutation_reference rhs_permute,
                                        const base_type& lhs,
                                        const base_type& rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs * rhs; };
    element_wise_op_(lambda, lhs_permute, rhs_permute, lhs, rhs);
}

TPARAMS
void EIGEN_TENSOR::permute_assignment_(const_permutation_reference rhs_permute,
                                       const base_type& rhs) {
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    if constexpr(Rank <= 1) {
        m_tensor_ = rhs_down->m_tensor_;
        return;
    } else {
        auto eigen_rhs_permute = to_eigen_permutation(rhs_permute);
        auto rhs_shuffled      = rhs_down->m_tensor_.shuffle(eigen_rhs_permute);
        m_tensor_              = rhs_shuffled;
    }
}

TPARAMS
void EIGEN_TENSOR::scalar_multiplication_(
  const_permutation_reference rhs_permute, FloatType scalar,
  const base_type& rhs) {
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    if constexpr(Rank <= 1) {
        m_tensor_ = rhs_down->m_tensor_ * scalar;
        return;
    } else {
        auto eigen_rhs_permute = to_eigen_permutation(rhs_permute);
        auto rhs_shuffled      = rhs_down->m_tensor_.shuffle(eigen_rhs_permute);
        m_tensor_              = rhs_shuffled * scalar;
    }
}

TPARAMS
template<typename OperationType>
void EIGEN_TENSOR::element_wise_op_(OperationType op,
                                    const_permutation_reference lhs_permute,
                                    const_permutation_reference rhs_permute,
                                    const base_type& lhs,
                                    const base_type& rhs) {
    const auto* lhs_down = dynamic_cast<const my_type*>(&lhs);
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    if constexpr(Rank <= 1) {
        m_tensor_ = op(lhs_down->m_tensor_, rhs_down->m_tensor_);
        return;
    } else {
        auto eigen_lhs_permute = to_eigen_permutation(lhs_permute);
        auto eigen_rhs_permute = to_eigen_permutation(rhs_permute);
        auto lhs_shuffled      = lhs_down->m_tensor_.shuffle(eigen_lhs_permute);
        auto rhs_shuffled      = rhs_down->m_tensor_.shuffle(eigen_rhs_permute);
        m_tensor_              = op(lhs_shuffled, rhs_shuffled);
    }
}

// template<typename TensorType>
// auto matrix_size(TensorType&& t, std::size_t row_ranks) {
//     std::size_t nrows = 1;
//     for(std::size_t i = 0; i < row_ranks; ++i) nrows *= t.extent(i);

//     std::size_t ncols = 1;
//     const auto rank   = t.rank();
//     for(std::size_t i = row_ranks; i < rank; ++i) ncols *= t.extent(i);
//     return std::make_pair(nrows, ncols);
// }

// TPARAMS
// void EIGEN_TENSOR::contraction_assignment_(label_type olabels,
//                                            label_type llabels,
//                                            label_type rlabels,
//                                            const_shape_reference
//                                            result_shape,
//                                            const_pimpl_reference lhs,
//                                            const_pimpl_reference rhs) {
//     ContractionPlanner plan(olabels, llabels, rlabels);

// #ifdef ENABLE_CUTENSOR
//     // Prepare m_tensor_
//     m_tensor_ = allocate_from_shape_(result_shape.as_smooth(),
//                                      std::make_index_sequence<Rank>());
//     m_tensor_.setZero();

//     // Dispatch to cuTENSOR
//     cutensor_contraction<my_type>(olabels, llabels, rlabels,
//     result_shape, lhs,
//                                   rhs, m_tensor_);
// #else
//     auto lt = lhs.clone();
//     auto rt = rhs.clone();
//     lt->permute_assignment(plan.lhs_permutation(), llabels, lhs);
//     rt->permute_assignment(plan.rhs_permutation(), rlabels, rhs);

//     const auto [lrows, lcols] = matrix_size(*lt, plan.lhs_free().size());
//     const auto [rrows, rcols] = matrix_size(*rt,
//     plan.rhs_dummy().size());

//     // Work out the types of the matrix amd a map
//     constexpr auto e_dyn       = ::Eigen::Dynamic;
//     constexpr auto e_row_major = ::Eigen::RowMajor;
//     using matrix_t = ::Eigen::Matrix<FloatType, e_dyn, e_dyn,
//     e_row_major>; using map_t    = ::Eigen::Map<matrix_t>;

//     eigen::data_type<FloatType, 2> buffer(lrows, rcols);

//     map_t lmatrix(lt->get_mutable_data(), lrows, lcols);
//     map_t rmatrix(rt->get_mutable_data(), rrows, rcols);
//     map_t omatrix(buffer.data(), lrows, rcols);
//     omatrix = lmatrix * rmatrix;

//     auto mlabels = plan.result_matrix_labels();
//     auto oshape  = result_shape(olabels);

//     // oshapes is the final shape, permute it to shape omatrix is
//     currently in auto temp_shape = result_shape.clone();
//     temp_shape->permute_assignment(mlabels, oshape);
//     auto mshape = temp_shape->as_smooth();

//     auto m_to_o = olabels.permutation(mlabels); // N.b. Eigen def is
//     inverse us

//     std::array<int, Rank> out_size;
//     std::array<int, Rank> m_to_o_array;
//     for(std::size_t i = 0; i < Rank; ++i) {
//         out_size[i]     = mshape.extent(i);
//         m_to_o_array[i] = m_to_o[i];
//     }

//     auto tensor = buffer.reshape(out_size);
//     if constexpr(Rank > 0) {
//         m_tensor_ = tensor.shuffle(m_to_o_array);
//     } else {
//         m_tensor_ = tensor;
//     }
// #endif
//     mark_for_rehash_();
// }

#undef EIGEN_TENSOR
#undef TPARAMS

#define DEFINE_EIGEN_TENSOR(TYPE)            \
    template class EigenTensorImpl<TYPE, 0>; \
    template class EigenTensorImpl<TYPE, 1>; \
    template class EigenTensorImpl<TYPE, 2>; \
    template class EigenTensorImpl<TYPE, 3>; \
    template class EigenTensorImpl<TYPE, 4>; \
    template class EigenTensorImpl<TYPE, 5>; \
    template class EigenTensorImpl<TYPE, 6>; \
    template class EigenTensorImpl<TYPE, 7>; \
    template class EigenTensorImpl<TYPE, 8>; \
    template class EigenTensorImpl<TYPE, 9>; \
    template class EigenTensorImpl<TYPE, 10>

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_EIGEN_TENSOR);

#undef DEFINE_EIGEN_TENSOR

} // namespace tensorwrapper::backends::eigen
