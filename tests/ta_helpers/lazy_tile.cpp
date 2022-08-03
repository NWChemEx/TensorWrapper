#include "tensorwrapper/ta_helpers/direct_tile.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include <catch2/catch.hpp>
#include <tiledarray.h>

template<typename TileType>
using ta_t = TA::DistArray<TileType, TA::SparsePolicy>;

using trange_t = TA::TiledRange;
using range_t  = TA::Range;
using tile_t   = TA::Tensor<double>;

/// Simple functor for filling in data tiles.
struct data_ftor {
    tile_t operator()(range_t range) { return tile_t(range, 1.0); }
};
using lazy_t = tensorwrapper::ta_helpers::LazyTile<tile_t, data_ftor>;

using tensorwrapper::ta_helpers::allclose;

TEST_CASE("LazyTile") {
    /// Inputs and comparison values
    auto& world = TA::get_default_world();
    auto trange = trange_t{{0, 3}, {0, 3}};
    auto I      = ta_t<tile_t>{world, trange};
    auto J      = ta_t<tile_t>{world, trange};
    ta_t<tile_t> Y;
    I.fill(1.0);
    J.fill(2.0);

    /// Assigns lazy tile to input tile
    auto tile_lambda = [](lazy_t& t, const range_t& r) -> float {
        t = lazy_t(r, data_ftor{});
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
}