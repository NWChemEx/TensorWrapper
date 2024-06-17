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

#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "../tensor/test_tensor.hpp"
#include "tensorwrapper/tensor/conversion/conversion.hpp"
#include "tensorwrapper/tensor/creation.hpp"
#include <catch2/catch.hpp>
#include <iostream>
#include <tensorwrapper/tensor/detail_/ta_to_tw.hpp>
#include <tiledarray.h>
#include <vector>
using std::cout;

using namespace tensorwrapper;
using namespace tensorwrapper::tensor;

TEST_CASE("TA_vs_TW_ADD", "[.ptest]") {
    const int kmatsize  = 1000;
    const int tile_size = 100;
    auto& world         = TA::get_default_world();
    using ta_type       = TA::TSpArrayD;
    using tw_type       = ScalarTensorWrapper;
    using tensorwrapper::tensor::detail_::ta_to_tw;
    // generate some random tensors
    std::vector<std::size_t> tile_boundaries;
    for(std::size_t i = 0; i <= kmatsize; i += tile_size)
        tile_boundaries.push_back(i);
    std::vector<TA::TiledRange1> ranges(
      2, TA::TiledRange1(tile_boundaries.begin(), tile_boundaries.end()));
    TA::TiledRange trange(ranges.begin(), ranges.end());

    ta_type lhs_ta(world, trange);
    ta_type rhs_ta(world, trange);
    lhs_ta.fill(0.5);
    rhs_ta.fill(0.5);

    auto lhs_tw = ta_to_tw(lhs_ta);
    auto rhs_tw = ta_to_tw(rhs_ta);

    using tensorwrapper::tensor::allclose;

    ta_type res_ta;
    tw_type res_tw;

    to_ta_distarrayd_t converter;
    REQUIRE(ta_helpers::allclose(converter.convert(lhs_tw.buffer()), lhs_ta));
    REQUIRE(ta_helpers::allclose(converter.convert(rhs_tw.buffer()), rhs_ta));

    // start benchmark

    BENCHMARK("TiledArray_add") {
        world.gop.fence();
        res_ta("i,j") = lhs_ta("i, j") + rhs_ta("i, j");
        world.gop.fence();
        return res_ta;
    };

    BENCHMARK("TensorWrapper_add") {
        world.gop.fence();
        res_tw("i,j") = lhs_tw("i, j") + rhs_tw("i, j");
        world.gop.fence();
        return res_tw;
    };
}
