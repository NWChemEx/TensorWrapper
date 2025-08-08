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

#include "tensorwrapper/ta_helpers/lazy_tile.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include <catch2/catch.hpp>
#include <tiledarray.h>

template<typename TileType>
using ta_t = TA::DistArray<TileType, TA::SparsePolicy>;

using trange_t = TA::TiledRange;
using range_t  = TA::Range;
using tile_t   = TA::Tensor<double>;

using lazy_t = tensorwrapper::ta_helpers::lazy_scalar_type;

using tensorwrapper::ta_helpers::allclose;

TEST_CASE("LazyTile") {
    /// Inputs and comparison values
    auto& world = TA::get_default_world();
    auto trange = trange_t{{0, 3}, {0, 3}};
    auto i      = tile_t{range_t{{0, 3}, {0, 3}}, 1.0};
    auto I      = ta_t<tile_t>{world, trange};
    auto J      = ta_t<tile_t>{world, trange};
    ta_t<tile_t> Y;
    I.fill(1.0);
    J.fill(2.0);

    /// Create data lambda and add to LazyTile evaluators
    auto data_lambda = [](range_t range) -> tile_t {
        return tile_t(range, 1.0);
    };
    lazy_t::add_evaluator(data_lambda, "test");

    /// Test lazy evaluation
    auto x = lazy_t(range_t{{0, 3}, {0, 3}}, "test");
    auto y = tile_t(x);
    REQUIRE(y == i);

    /// Assigns lazy tile to input tile
    auto tile_lambda = [](lazy_t& t, const range_t& r) -> float {
        t = lazy_t(r, "test");
        return 1.0;
    };

    /// Make a lazy array for testing
    auto X = TiledArray::make_array<ta_t<lazy_t>>(world, trange, tile_lambda);

    /// Test operations with the lazy array
    Y("i,j") = X("i,j");
    REQUIRE(allclose(Y, I));

    Y("i,j") = I("i,j") + X("i,j");
    REQUIRE(allclose(Y, J));

    /// This will not compile because you can't assign to a lazy tile
    // X("i,j") = I("i,j");

    /// Test clone
    auto clone = TA::clone(X);
    Y("i,j")   = clone("i,j");
    REQUIRE(allclose(Y, I));
}
