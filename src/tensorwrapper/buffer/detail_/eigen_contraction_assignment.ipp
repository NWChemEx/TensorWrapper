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
#pragma once
#include "../../backends/eigen.hpp"
#include "../contraction_planner.hpp"
#include "eigen_tensor.hpp"

namespace tensorwrapper::buffer::detail_ {

template<typename TensorType>
auto matrix_size(TensorType&& t, std::size_t row_ranks) {
    std::size_t nrows = 1;
    for(std::size_t i = 0; i < row_ranks; ++i) nrows *= t.extent(i);

    std::size_t ncols = 1;
    const auto rank   = t.rank();
    for(std::size_t i = row_ranks; i < rank; ++i) ncols *= t.extent(i);
    return std::make_pair(nrows, ncols);
}

template<typename FloatType, unsigned int Rank>
void EigenTensor<FloatType, Rank>::contraction_assignment_(
  label_type olabels, label_type llabels, label_type rlabels,
  const_shape_reference result_shape, const_pimpl_reference lhs,
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

} // namespace tensorwrapper::buffer::detail_