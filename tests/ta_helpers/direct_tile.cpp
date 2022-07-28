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
using lazy_t = LazyTile<tile_t, data_ftor>;

using tensorwrapper::ta_helpers::allclose;

TEST_CASE("LazyTile") {
    auto& world = TA::get_default_world();

    trange_t trange{{0, 3}, {0, 3}};
    ta_t<tile_t> I{world, trange}, J{world, trange}, Y;
    I.fill(1.0);
    J.fill(2.0);

    auto tile_lambda = [](lazy_t& t, const range_t& r) -> float {
        t = lazy_t(r, data_ftor{});
        return 1.0;
    };
    auto X = TiledArray::make_array<ta_t<lazy_t>>(world, trange, tile_lambda);

    /// Test operations with the lazy array
    Y("i,j") = X("i,j");
    REQUIRE(allclose(Y, I));

    Y("i,j") = I("i,j") + X("i,j");
    REQUIRE(allclose(Y, J));

    /// This will not compile because you can't assign to a lazy tile
    // X("i,j") = I("i,j");
}