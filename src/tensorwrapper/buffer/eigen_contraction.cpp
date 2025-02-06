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

#include "contraction_planner.hpp"
#include "eigen_contraction.hpp"
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>

namespace tensorwrapper::buffer {

using rank_type               = unsigned short;
using return_type             = BufferBase::dsl_reference;
using label_type              = BufferBase::label_type;
using const_labeled_reference = BufferBase::const_labeled_reference;

static constexpr unsigned int max_rank = 10;

template<typename FloatType, rank_type Rank, typename TensorType>
FloatType* get_data(TensorType& tensor) {
    using allocator_type = allocator::Eigen<FloatType, Rank>;
    if constexpr(Rank > max_rank) {
        const auto sr  = std::to_string(max_rank);
        const auto msg = "Tensors with rank > " + sr + " are not supported";
        throw std::runtime_error(msg);
    } else {
        if(tensor.layout().rank() == Rank) {
            return allocator_type::rebind(tensor).value().data();
        } else {
            return get_data<FloatType, Rank + 1>(tensor);
        }
    }
}

template<typename TensorType>
auto matrix_size(TensorType&& t, std::size_t row_ranks) {
    const auto shape  = t.layout().shape().as_smooth();
    std::size_t nrows = 1;
    for(std::size_t i = 0; i < row_ranks; ++i) nrows *= shape.extent(i);

    std::size_t ncols = 1;
    const auto rank   = shape.rank();
    for(std::size_t i = row_ranks; i < rank; ++i) ncols *= shape.extent(i);
    return std::make_pair(nrows, ncols);
}

template<typename FloatType, rank_type Rank>
return_type eigen_contraction(Eigen<FloatType, Rank>& result,
                              label_type olabels, const_labeled_reference lhs,
                              const_labeled_reference rhs) {
    const auto& llabels = lhs.labels();
    const auto& lobject = lhs.object();
    const auto& rlabels = rhs.labels();
    const auto& robject = rhs.object();

    ContractionPlanner plan(olabels, llabels, rlabels);
    auto lt = lobject.clone();
    auto rt = robject.clone();
    lt->permute_assignment(plan.lhs_permutation(), lhs);
    rt->permute_assignment(plan.rhs_permutation(), rhs);
    const auto [lrows, lcols] = matrix_size(*lt, plan.lhs_free().size());
    const auto [rrows, rcols] = matrix_size(*rt, plan.rhs_dummy().size());

    // Work out the types of the matrix amd a map
    constexpr auto e_dyn       = ::Eigen::Dynamic;
    constexpr auto e_row_major = ::Eigen::RowMajor;
    using matrix_t = ::Eigen::Matrix<FloatType, e_dyn, e_dyn, e_row_major>;
    using map_t    = ::Eigen::Map<matrix_t>;

    typename Eigen<FloatType, 2>::data_type buffer(lrows, rcols);

    map_t lmatrix(get_data<FloatType, 0>(*lt), lrows, lcols);
    map_t rmatrix(get_data<FloatType, 0>(*rt), rrows, rcols);
    map_t omatrix(buffer.data(), lrows, rcols);
    omatrix = lmatrix * rmatrix;

    auto mlabels = plan.result_matrix_labels();
    auto oshape  = result.layout().shape()(olabels);

    // oshapes is the final shape, permute it to shape omatrix is currently in
    auto temp_shape = result.layout().shape().clone();
    temp_shape->permute_assignment(mlabels, oshape);
    auto mshape = temp_shape->as_smooth();

    auto m_to_o = olabels.permutation(mlabels); // N.b. Eigen def is inverse us

    std::array<std::size_t, Rank> out_size;
    std::array<std::size_t, Rank> m_to_o_array;
    for(std::size_t i = 0; i < Rank; ++i) {
        out_size[i]     = mshape.extent(i);
        m_to_o_array[i] = m_to_o[i];
    }

    auto tensor = buffer.reshape(out_size);
    if constexpr(Rank > 0) {
        result.value() = tensor.shuffle(m_to_o_array);
    } else {
        result.value() = tensor;
    }
    return result;
}

#define EIGEN_CONTRACTION_(FLOAT_TYPE, RANK)               \
    template return_type eigen_contraction(                \
      Eigen<FLOAT_TYPE, RANK>& result, label_type olabels, \
      const_labeled_reference lhs, const_labeled_reference rhs)

#define EIGEN_CONTRACTION(FLOAT_TYPE)  \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 0); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 1); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 2); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 3); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 4); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 5); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 6); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 7); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 8); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 9); \
    EIGEN_CONTRACTION_(FLOAT_TYPE, 10)

EIGEN_CONTRACTION(float);
EIGEN_CONTRACTION(double);

#undef EIGEN_CONTRACTION
#undef EIGEN_CONTRACTION_
} // namespace tensorwrapper::buffer