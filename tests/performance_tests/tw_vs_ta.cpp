#define CATCH_CONFIG_ENABLE_BENCHMARKING
#include "../tensor/test_tensor.hpp"
#include "tensorwrapper/tensor/creation.hpp"
#include <catch2/catch.hpp>
#include <iostream>
#include <tensorwrapper/tensor/detail_/ta_to_tw.hpp>
#include <tiledarray.h>
#include <vector>

using namespace tensorwrapper;
using namespace tensorwrapper::tensor;

TEST_CASE("TA_vs_TW", "[.][ptest]") {
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

    REQUIRE(ta_helpers::allclose(lhs_tw.get<ta_type>(), lhs_ta));
    REQUIRE(ta_helpers::allclose(rhs_tw.get<ta_type>(), rhs_ta));

    // start benchmark
    BENCHMARK("TiledArray_mult") {
        world.gop.fence();
        return res_ta("i,j") = lhs_ta("i, k") * rhs_ta("k, j");
        world.gop.fence();
    };

    BENCHMARK("TensorWrapper_mult") {
        world.gop.fence();
        return res_tw("i,j") = lhs_tw("i, k") * rhs_tw("k, j");
        world.gop.fence();
    };

    BENCHMARK("TiledArray_add") {
        world.gop.fence();
        return res_ta("i,j") = lhs_ta("i, j") + rhs_ta("i, j");
        world.gop.fence();
    };

    BENCHMARK("TensorWrapper_add") {
        world.gop.fence();
        res_tw("i,j") = lhs_tw("i, j") + rhs_tw("i, j");
        world.gop.fence();
        return res_tw;
    };
}