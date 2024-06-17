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

#include "../shapes/make_tot_shape.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include <catch2/catch.hpp>
#include <tensorwrapper/tensor/approximately_equal.hpp>
#include <tensorwrapper/tensor/backends/tiledarray.hpp>

using namespace tensorwrapper::tensor;
using namespace tensorwrapper::tensor::backends;

using tat_t   = TA::TSpArrayD;
using tot_t   = TA::TSpArray<TA::Tensor<double>>;
using tile_t  = typename tot_t::value_type;
using inner_t = typename tile_t::value_type;

using tensorwrapper::ta_helpers::allclose;
using tensorwrapper::ta_helpers::allclose_tot;

TEST_CASE("Tiled Array Backend") {
    auto& world = TA::get_default_world();

    /// TA Values
    inner_t v0(TA::Range{3}, {1.0, 2.0, 3.0});
    tat_t corr_mat_ta(world, {{1.0, 2.0}, {3.0, 4.0}});
    tot_t corr_vov_ta(world, {v0, v0, v0});

    /// ToT building blocks
    auto l = [](const auto& outer_idx, const auto& lo, const auto& up,
                auto* data) {
        for(auto i = lo[0]; i < up[0]; ++i) data[i] = i + 1;
    };
    auto alloc = default_allocator<field::Tensor>();
    auto shape = testing::make_uniform_tot_shape({3}, {3});

    /// TW Values
    ScalarTensorWrapper corr_mat_tw({{1.0, 2.0}, {3.0, 4.0}});
    TensorOfTensorsWrapper corr_vov_tw(l, shape.clone(), alloc->clone());

    /// Check tolerances
    auto rtol = 1e-10;
    auto atol = 1e-8;

    SECTION("wrap_ta") {
        SECTION("Scalar") {
            auto wrapped = wrap_ta(corr_mat_ta);
            REQUIRE(are_approximately_equal(wrapped, corr_mat_tw, rtol, atol));
        }
        SECTION("Tensor of Tensor") {
            auto wrapped = wrap_ta(corr_vov_ta);
            REQUIRE(are_approximately_equal(wrapped, corr_vov_tw, rtol, atol));
        }
    }

    SECTION("unwrap_ta") {
        SECTION("Scalar") {
            SECTION("non-const") {
                auto& unwrapped = unwrap_ta(corr_mat_tw);
                REQUIRE(allclose(unwrapped, corr_mat_ta));
            }
            SECTION("const") {
                const auto& unwrapped = unwrap_ta(corr_mat_tw);
                REQUIRE(allclose(unwrapped, corr_mat_ta));
            }
        }
        SECTION("Tensor of Tensor") {
            SECTION("non-const") {
                auto& unwrapped = unwrap_ta(corr_vov_tw);
                REQUIRE(allclose_tot(unwrapped, corr_vov_ta, 1));
            }
            SECTION("const") {
                const auto& unwrapped = unwrap_ta(corr_vov_tw);
                REQUIRE(allclose_tot(unwrapped, corr_vov_ta, 1));
            }
        }
    }
}
