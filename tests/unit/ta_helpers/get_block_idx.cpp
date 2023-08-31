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

#include "tensorwrapper/ta_helpers/get_block_idx.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::ta_helpers;

TEST_CASE("get_block_idx") {
    auto& world = TA::get_default_world();
    SECTION("Vector") {
        using vector_il = TA::detail::vector_il<int>;
        TA::TiledRange trange{{0, 1, 2}};
        TA::TSpArrayD t(world, trange, vector_il{0, 1});
        for(long i = 0; i < 2; ++i) {
            auto idx = get_block_idx(t, t.find(i).get());
            decltype(idx) corr;
            corr.push_back(i);
            REQUIRE(idx == corr);
        }
    }

    SECTION("Matrix") {
        using vector_il = TA::detail::vector_il<int>;
        using matrix_il = TA::detail::matrix_il<int>;
        TA::TiledRange trange{{0, 1, 2}, {0, 1, 2}};
        TA::TSpArrayD t(world, trange,
                        matrix_il{vector_il{0, 1}, vector_il{2, 3}});
        for(long i = 0; i < 2; ++i) {
            for(long j = 0; j < 2; ++j) {
                auto idx = get_block_idx(t, t.find({i, j}).get());
                decltype(idx) corr{i, j};
                REQUIRE(idx == corr);
            }
        }
    }

    SECTION("Tensor") {
        using vector_il = TA::detail::vector_il<int>;
        using matrix_il = TA::detail::matrix_il<int>;
        using tensor_il = TA::detail::tensor3_il<int>;
        TA::TiledRange trange{{0, 1, 2}, {0, 1, 2}, {0, 1, 2}};
        TA::TSpArrayD t(world, trange,
                        tensor_il{matrix_il{vector_il{0, 1}, vector_il{2, 3}},
                                  matrix_il{vector_il{4, 5}, vector_il{6, 7}}});
        for(long i = 0; i < 2; ++i) {
            for(long j = 0; j < 2; ++j) {
                for(long k = 0; k < 2; ++k) {
                    auto idx = get_block_idx(t, t.find({i, j, k}).get());
                    decltype(idx) corr{i, j, k};
                    REQUIRE(idx == corr);
                }
            }
        }
    }
}
