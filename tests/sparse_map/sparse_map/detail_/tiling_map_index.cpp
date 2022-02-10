#include "tensorwrapper/sparse_map/sparse_map/detail_/tiling_map_index.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::sparse_map;
using namespace tensorwrapper::sparse_map::detail_;

TEST_CASE("Tiling and Untiling SparseMaps") {
    // Some indices
    Index i0{0}, i1{1}, i2{2}, i3{3}, i4{4}, i5{5};

    // Make SparseMaps with the expected form
    SparseMap eesm{{i0, {i0, i1}}, {i1, {i0, i1}}, {i2, {i2, i3}},
                   {i3, {i2, i3}}, {i4, {i4, i5}}, {i5, {i4, i5}}};
    SparseMap etsm{{i0, {i0}}, {i1, {i0}}, {i2, {i1}},
                   {i3, {i1}}, {i4, {i2}}, {i5, {i2}}};
    SparseMap tesm{{i0, {i0, i1}}, {i1, {i2, i3}}, {i2, {i4, i5}}};
    SparseMap ttsm{{i0, {i0}}, {i1, {i1}}, {i2, {i2}}};

    // TiledRange
    TA::TiledRange tr{{0, 2, 4, 6}};

    SECTION("tile_independent_indices") {
        auto new_sm = tile_independent_indices(eesm, tr);
        REQUIRE(new_sm == tesm);
    }

    SECTION("tile_dependent_indices") {
        auto new_sm = tile_dependent_indices(eesm, tr);
        REQUIRE(new_sm == etsm);
    }

    SECTION("tile_indices") {
        auto new_sm = tile_indices(eesm, tr, tr);
        REQUIRE(new_sm == ttsm);
    }

    SECTION("untile_independent_indices") {
        auto new_sm = untile_independent_indices(ttsm, tr);
        REQUIRE(new_sm == etsm);
    }

    SECTION("untile_dependent_indices") {
        auto new_sm = untile_dependent_indices(ttsm, tr);
        REQUIRE(new_sm == tesm);
    }

    SECTION("untile_indices") {
        auto new_sm = untile_indices(ttsm, tr, tr);
        REQUIRE(new_sm == eesm);
    }
}

TEST_CASE("Tiling a Domain") {
    // Some indices
    Index i0{0}, i1{1}, i2{2}, i3{3}, i4{4}, i5{5};
    Domain d0{i0, i1, i2, i3, i4, i5}, d1{i0, i1, i2};

    // TiledRange
    TA::TiledRange tr{{0, 2, 4, 6}};

    auto new_d = tile_domain(d0, tr);
    REQUIRE(new_d == d1);
}