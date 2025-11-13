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

#include "../../buffer/contraction_planner.hpp"
#include "eigen_tensor_impl.hpp"
#include <iomanip>
#include <sstream>

namespace tensorwrapper::backends::eigen {

#define TPARAMS template<typename FloatType, unsigned int Rank>
#define EIGEN_TENSOR EigenTensorImpl<FloatType, Rank>

TPARAMS
EIGEN_TENSOR::EigenTensorImpl(std::span<value_type> data,
                              const_shape_reference shape) :
  m_tensor_(make_from_shape_(data, shape, std::make_index_sequence<Rank>())) {}

TPARAMS
auto EIGEN_TENSOR::permuted_copy_(label_type out, label_type in) const
  -> permuted_copy_return_type {
    using value_type = FloatType;
    std::vector<value_type> buffer(this->size(), value_type{0});
    std::span<value_type> buffer_span(buffer.data(), buffer.size());

    // Make a shape::Smooth object for tensor
    std::vector<std::size_t> old_shape_vec(this->rank());
    for(std::size_t i = 0; i < old_shape_vec.size(); ++i) {
        old_shape_vec[i] = this->extent(i);
    }
    shape_type old_shape(old_shape_vec.begin(), old_shape_vec.end());
    shape_type new_shape(old_shape);
    new_shape(out) = old_shape(in);
    auto pnew_tensor =
      std::make_unique<EigenTensorImpl>(buffer_span, new_shape);
    pnew_tensor->permute_assignment(out, in, *this);
    return std::make_pair(std::move(buffer), std::move(pnew_tensor));
}

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
void EIGEN_TENSOR::addition_assignment_(label_type this_label,
                                        label_type lhs_label,
                                        label_type rhs_label,
                                        const base_type& lhs,
                                        const base_type& rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs + rhs; };
    element_wise_op_(lambda, this_label, lhs_label, rhs_label, lhs, rhs);
}

TPARAMS
void EIGEN_TENSOR::subtraction_assignment_(label_type this_label,
                                           label_type lhs_label,
                                           label_type rhs_label,
                                           const base_type& lhs,
                                           const base_type& rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs - rhs; };
    element_wise_op_(lambda, this_label, lhs_label, rhs_label, lhs, rhs);
}

TPARAMS
void EIGEN_TENSOR::hadamard_assignment_(label_type this_label,
                                        label_type lhs_label,
                                        label_type rhs_label,
                                        const base_type& lhs,
                                        const base_type& rhs) {
    auto lambda = [](auto&& lhs, auto&& rhs) { return lhs * rhs; };
    element_wise_op_(lambda, this_label, lhs_label, rhs_label, lhs, rhs);
}

TPARAMS
void EIGEN_TENSOR::permute_assignment_(label_type this_label,
                                       label_type rhs_label,
                                       const base_type& rhs) {
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    if constexpr(Rank <= 1) {
        m_tensor_ = rhs_down->m_tensor_;
        return;
    } else {
        if(this_label != rhs_label) { // We need to permute rhs first
            // Eigen adopts the opposite definition of permutation from us.
            auto r_to_l = this_label.permutation(rhs_label);
            // Eigen wants int objects
            std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
            m_tensor_ = rhs_down->m_tensor_.shuffle(r_to_l2);
        } else {
            m_tensor_ = rhs_down->m_tensor_;
        }
    }
}

TPARAMS
void EIGEN_TENSOR::scalar_multiplication_(label_type this_label,
                                          label_type rhs_label,
                                          FloatType scalar,
                                          const base_type& rhs) {
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    if constexpr(Rank <= 1) {
        m_tensor_ = rhs_down->m_tensor_ * scalar;
        return;
    } else {
        if(this_label != rhs_label) { // We need to permute rhs first
            auto r_to_l = rhs_label.permutation(this_label);
            // Eigen wants int objects
            std::vector<int> r_to_l2(r_to_l.begin(), r_to_l.end());
            m_tensor_ = rhs_down->m_tensor_.shuffle(r_to_l2) * scalar;
        } else {
            m_tensor_ = rhs_down->m_tensor_ * scalar;
        }
    }
}

TPARAMS
template<typename OperationType>
void EIGEN_TENSOR::element_wise_op_(OperationType op, label_type this_label,
                                    label_type lhs_label, label_type rhs_label,
                                    const base_type& lhs,
                                    const base_type& rhs) {
    const auto* lhs_down = dynamic_cast<const my_type*>(&lhs);
    const auto* rhs_down = dynamic_cast<const my_type*>(&rhs);

    // Whose indices match whose?
    bool this_matches_lhs = (this_label == lhs_label);
    bool this_matches_rhs = (this_label == rhs_label);
    bool lhs_matches_rhs  = (lhs_label == rhs_label);

    // The three possible permutations we may need to apply
    auto get_permutation = [](auto&& lhs_, auto&& rhs_) {
        auto l_to_r = lhs_.permutation(rhs_);
        return std::vector<int>(l_to_r.begin(), l_to_r.end());
    };
    auto r_to_l    = get_permutation(rhs_label, lhs_label);
    auto l_to_r    = get_permutation(lhs_label, rhs_label);
    auto this_to_r = get_permutation(this_label, rhs_label);

    auto& lhs_eigen = lhs_down->m_tensor_;
    auto& rhs_eigen = rhs_down->m_tensor_;

    if constexpr(Rank <= 1) {
        m_tensor_ = op(lhs_eigen, rhs_eigen);
        return;
    } else {
        if(this_matches_lhs && this_matches_rhs) { // No permutations
            m_tensor_ = op(lhs_eigen, rhs_eigen);
        } else if(this_matches_lhs) { // RHS needs permuted
            m_tensor_ = op(lhs_eigen, rhs_eigen.shuffle(r_to_l));
        } else if(this_matches_rhs) { // LHS needs permuted
            m_tensor_ = op(lhs_eigen.shuffle(l_to_r), rhs_eigen);
        } else if(lhs_matches_rhs) { // This needs permuted
            m_tensor_ = op(lhs_eigen, rhs_eigen).shuffle(this_to_r);
        } else { // Everything needs permuted
            auto lhs_shuffled = lhs_eigen.shuffle(l_to_r);
            m_tensor_         = op(lhs_shuffled, rhs_eigen).shuffle(this_to_r);
        }
    }
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
void EIGEN_TENSOR::contraction_assignment_(label_type this_label,
                                           label_type lhs_label,
                                           label_type rhs_label,
                                           const base_type& lhs,
                                           const base_type& rhs) {
    buffer::ContractionPlanner plan(this_label, lhs_label, rhs_label);

    // Transpose, Transpose part of TTGT
    auto&& [new_lhs_buffer, pnew_lhs_tensor] =
      lhs.permuted_copy(plan.lhs_permutation(), lhs_label);

    auto&& [new_rhs_buffer, pnew_rhs_tensor] =
      rhs.permuted_copy(plan.rhs_permutation(), rhs_label);

    // Gemm part of TTGT
    auto olabels = plan.result_matrix_labels();

    auto&& [out_buffer, pout_tensor] = this->permuted_copy(olabels, this_label);

    const auto [lrows, lcols] =
      matrix_size(*pnew_lhs_tensor, plan.lhs_free().size());
    const auto [rrows, rcols] =
      matrix_size(*pnew_rhs_tensor, plan.rhs_dummy().size());

    // Work out the types of the matrix amd a map
    constexpr auto e_dyn       = ::Eigen::Dynamic;
    constexpr auto e_row_major = ::Eigen::RowMajor;
    using matrix_t = ::Eigen::Matrix<FloatType, e_dyn, e_dyn, e_row_major>;
    using map_t    = ::Eigen::Map<matrix_t>;

    map_t lmatrix(new_lhs_buffer.data(), lrows, lcols);
    map_t rmatrix(new_rhs_buffer.data(), rrows, rcols);
    map_t omatrix(out_buffer.data(), lrows, rcols);

    omatrix = lmatrix * rmatrix;

    // The last transpose part of TTGT
    this->permute_assignment(this_label, olabels, *pout_tensor);
}

#undef EIGEN_TENSOR
#undef TPARAMS

std::unique_ptr<EigenTensor<FloatType>> make_eigen_tensor(
  std::span<FloatType> data, shape::SmoothView<const shape::Smooth> shape) {
    switch(shape.rank()) {
        case 0:
            return std::make_unique<EigenTensorImpl<FloatType, 0>>(data, shape);
        case 1:
            return std::make_unique<EigenTensorImpl<FloatType, 1>>(data, shape);
        case 2:
            return std::make_unique<EigenTensorImpl<FloatType, 2>>(data, shape);
        case 3:
            return std::make_unique<EigenTensorImpl<FloatType, 3>>(data, shape);
        case 4:
            return std::make_unique<EigenTensorImpl<FloatType, 4>>(data, shape);
        case 5:
            return std::make_unique<EigenTensorImpl<FloatType, 5>>(data, shape);
        case 6:
            return std::make_unique<EigenTensorImpl<FloatType, 6>>(data, shape);
        case 7:
            return std::make_unique<EigenTensorImpl<FloatType, 7>>(data, shape);
        case 8:
            return std::make_unique<EigenTensorImpl<FloatType, 8>>(data, shape);
        case 9:
            return std::make_unique<EigenTensorImpl<FloatType, 9>>(data, shape);
        case 10:
            return std::make_unique<EigenTensorImpl<FloatType, 10>>(data,
                                                                    shape);
        default:
            throw std::runtime_error(
              "EigenTensor backend only supports ranks 0 through 10.");
    }
}

#define DEFINE_MAKE_EIGEN_TENSOR(TYPE)                                   \
    template std::unique_ptr<EigenTensor<TYPE>> make_eigen_tensor<TYPE>( \
      std::span<TYPE> data, shape::SmoothView<const shape::Smooth> shape);

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_MAKE_EIGEN_TENSOR);

#undef DEFINE_MAKE_EIGEN_TENSOR

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
