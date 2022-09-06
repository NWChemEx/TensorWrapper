/*
 * Copyright 2022 NWChemEx-Project
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

#include <catch2/catch.hpp>
#include <tensorwrapper/ta_helpers/submatrix_functions.hpp>
#include <tensorwrapper/ta_helpers/ta_helpers.hpp>

using tensor_type = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;
using tensorwrapper::ta_helpers::allclose;
using tensorwrapper::ta_helpers::expand_submatrix;
using tensorwrapper::ta_helpers::submatrix;

TEST_CASE("Submatrix Functions") {
    auto& world = TA::get_default_world();

    // Make inputs and comparison values
    TA::TiledRange1 tr1{0, 1, 2, 3};
    TA::TiledRange1 tr2{0, 1, 2};

    TA::TiledRange trange1{tr1, tr1};
    TA::TiledRange trange2{tr2, tr2};

    TA::Tensor<float> mask(trange1.tiles_range(), 0.0);
    mask(0, 0) = std::numeric_limits<float>::max();
    mask(0, 2) = std::numeric_limits<float>::max();
    mask(2, 0) = std::numeric_limits<float>::max();
    mask(2, 2) = std::numeric_limits<float>::max();
    TA::SparseShape<float> shape(mask, trange1);

    auto full_matrix = TA::diagonal_array<tensor_type>(world, trange1, 1.0);
    auto corr_submat = TA::diagonal_array<tensor_type>(world, trange2, 1.0);

    tensor_type corr_matrix;
    corr_matrix("i, j") = full_matrix("i, j").set_shape(shape);

    // Test functions
    auto submat = submatrix<double>(full_matrix, mask);
    auto matrix = expand_submatrix<double>(corr_submat, trange1, mask);

    REQUIRE(allclose(matrix, corr_matrix));
    REQUIRE(allclose(submat, corr_submat));
}
