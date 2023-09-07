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

#include "tensorwrapper/ta_helpers/remove_redundancy.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::ta_helpers;

using tensor_type = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;
using vector_t    = std::initializer_list<double>;
using matrix_t    = std::initializer_list<vector_t>;
using vector_il   = TA::detail::vector_il<double>;
using matrix_il   = TA::detail::matrix_il<double>;

namespace {

// Correct overlap of redundant_PAOs_corr
constexpr matrix_t redundant_PAO_overlap_corr{{0.41629351, -0.34700249},
                                              {-0.34700249, 0.41629351}};

// Correct redundant PAOs for C_data and S_data
constexpr matrix_t redundant_PAOs_corr{{0.381648, -0.618352},
                                       {-0.618352, 0.381648}};

// Correct normalized, non-redundant PAOs for C_data and S_data
constexpr matrix_t NRC_corr_data{{0.6358462574920218, -0.8093539841320376},
                                 {0.6358462574920218, 0.8093539841320376}};

} // namespace

TEST_CASE("remove_redundancy") {
    auto& world = TA::get_default_world();
    tensor_type CTilde_corr(world, redundant_PAOs_corr);
    tensor_type STilde_corr(world, redundant_PAO_overlap_corr);

    SECTION("No redundancy") {
        auto NRC = remove_redundancy(CTilde_corr, STilde_corr);
        tensor_type NRC_corr(world, NRC_corr_data);
        REQUIRE(allclose(NRC, NRC_corr));
    }

    SECTION("One redundancy") {
        auto NRC = remove_redundancy(CTilde_corr, STilde_corr, 0.1);
        tensor_type NRC_corr(world, matrix_il{vector_il{-0.8093539841320377},
                                              vector_il{0.8093539841320377}});
        REQUIRE(allclose(NRC, NRC_corr));
    }
}

TEST_CASE("sparse_remove_redundancy") {
    using tile_type   = TA::TensorD;
    using tensor_type = TA::DistArray<TA::Tensor<tile_type>>;
    auto& world       = TA::get_default_world();
    using tvector_il  = TA::detail::vector_il<tile_type>;

    TA::Range r(std::vector<int>{2, 2});
    tile_type S0(r, {0.41629351, -0.34700249, -0.34700249, 0.41629351});
    tensor_type S(world, tvector_il{S0, S0});
    tile_type C0(r, {0.381648, -0.618352, -0.618352, 0.381648});
    tensor_type C(world, tvector_il{C0, C0});

    SECTION("No redundancy") {
        auto NRC = sparse_remove_redundancy(C, S);
        tile_type corr_tile(r,
                            vector_il{0.6358462574920218, -0.8093539841320376,
                                      0.6358462574920218, 0.8093539841320376});
        tensor_type corr(world, tvector_il{corr_tile, corr_tile});
        REQUIRE(allclose_tot(NRC, corr, 2));
    }

    SECTION("One redundancy") {
        auto NRC = sparse_remove_redundancy(C, S, 0.1);
        tile_type corr_tile(
          r, {0.0, -0.8093539841320376, 0.0, 0.8093539841320376});
        tensor_type corr(world, tvector_il{corr_tile, corr_tile});
        REQUIRE(allclose_tot(NRC, corr, 2));
    }
}
