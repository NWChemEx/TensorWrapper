#include "../buffer/make_pimpl.hpp"
#include "tensorwrapper/tensor/allocators/allocators.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;

TEMPLATE_TEST_CASE("TiledArrayAllocator Generic", "[allocator][ta]",
                   field::Scalar, field::Tensor) {
    using field_type = TestType;
    using alloc_type = allocator::TiledArrayAllocator<field_type>;
    using allocator::ta::Distribution;
    using allocator::ta::Storage;
    using allocator::ta::Tiling;

    alloc_type defaulted;
    alloc_type non_default(Storage::Core, Tiling::SingleElementTile,
                           Distribution::Distributed);

    SECTION("Default State") {
        REQUIRE(defaulted.storage() == Storage::Core);
        REQUIRE(defaulted.tiling() == Tiling::OneBigTile);
        REQUIRE(defaulted.dist() == Distribution::Replicated);
    }

    SECTION("Manual Ctor") {
        REQUIRE(non_default.storage() == Storage::Core);
        REQUIRE(non_default.tiling() == Tiling::SingleElementTile);
        REQUIRE(non_default.dist() == Distribution::Distributed);

        REQUIRE(defaulted != non_default);
        REQUIRE_FALSE(defaulted == non_default);
    }

    SECTION("Copy Ctor") {
        alloc_type copy(defaulted);
        REQUIRE(defaulted == copy);
        REQUIRE_FALSE(defaulted != copy);
    }

    SECTION("Move Ctor") {
        alloc_type other_defaulted;
        alloc_type copy(std::move(other_defaulted));
        REQUIRE(defaulted == copy);
        REQUIRE_FALSE(defaulted != copy);
    }

    SECTION("is_equal") {
        SECTION("Both Default") {
            alloc_type other_defaulted;
            REQUIRE(defaulted.is_equal(other_defaulted));
            REQUIRE(other_defaulted.is_equal(defaulted));
        }

        SECTION("Different Specs") {
            REQUIRE_FALSE(non_default.is_equal(defaulted));
            REQUIRE_FALSE(defaulted.is_equal(non_default));
        }
    }

    SECTION("Clone") {
        auto copy = defaulted.clone();
        REQUIRE(copy->is_equal(defaulted));
    }
}

TEST_CASE("TiledArrayAllocator<Scalar>") {
    using field_type     = field::Scalar;
    using buffer_type    = buffer::Buffer<field_type>;
    using allocator_type = allocator::TiledArrayAllocator<field_type>;
    using allocator::ta::Distribution;
    using allocator::ta::Storage;
    using allocator::ta::Tiling;

    using extents_type = typename allocator_type::extents_type;
    using shape_type   = typename allocator_type::shape_type;

    auto&& [pvec, pmat, pt3d] = testing::make_pimpl<field_type>();
    buffer_type vec(pvec->clone());
    buffer_type mat(pmat->clone());
    buffer_type ten(pt3d->clone());

    extents_type vec_extents{3};
    extents_type mat_extents{2, 2};
    extents_type ten_extents{2, 2, 2};

    shape_type vec_shape(vec_extents);
    shape_type mat_shape(mat_extents);
    shape_type ten_shape(ten_extents);

    SECTION("OneBigTile") {
        allocator_type alloc(Storage::Core, Tiling::OneBigTile);

        SECTION("allocate(rank 1)") {
            size_t inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++; // Count the number of invocations
                REQUIRE(lo.size() == 1);
                REQUIRE(up.size() == 1);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 3);
                REQUIRE(lo[0] < up[0]);
                size_t extent = up[0] - lo[0];
                for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
            };

            auto buf = alloc.allocate(fxn, vec_shape);
            REQUIRE(inner_tile_count == 1); // OneBigTile has only 1 tile
            REQUIRE(buf == vec);
        }

        SECTION("allocate(rank 2)") {
            size_t inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++; // Count the number of invocations
                REQUIRE(lo.size() == 2);
                REQUIRE(up.size() == 2);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 2);
                REQUIRE(lo[0] < up[0]);
                REQUIRE(lo[1] >= 0);
                REQUIRE(up[1] <= 2);
                REQUIRE(lo[1] < up[1]);
                size_t extent_0 = up[0] - lo[0];
                size_t extent_1 = up[1] - lo[1];
                for(auto i = 0; i < extent_0; ++i)
                    for(auto j = 0; j < extent_1; ++j) {
                        data[i * extent_1 + j] =
                          (i + lo[0]) * 2 + (j + lo[1]) + 1;
                    }
            };

            auto buf = alloc.allocate(fxn, mat_shape);
            REQUIRE(inner_tile_count == 1); // OneBigTile has only 1 tile
            REQUIRE(buf == mat);
        }

        SECTION("allocate(rank 3)") {
            size_t inner_tile_count = 0;
            auto fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                           double* data) {
                inner_tile_count++;
                REQUIRE(lo.size() == 3);
                REQUIRE(up.size() == 3);
                REQUIRE(lo[0] >= 0);
                REQUIRE(up[0] <= 2);
                REQUIRE(lo[0] < up[0]);
                REQUIRE(lo[1] >= 0);
                REQUIRE(up[1] <= 2);
                REQUIRE(lo[1] < up[1]);
                REQUIRE(lo[2] >= 0);
                REQUIRE(up[2] <= 2);
                REQUIRE(lo[2] < up[2]);
                size_t extent_0 = up[0] - lo[0];
                size_t extent_1 = up[1] - lo[1];
                size_t extent_2 = up[2] - lo[2];
                for(auto i = 0; i < extent_0; ++i)
                    for(auto j = 0; j < extent_1; ++j)
                        for(auto k = 0; k < extent_2; ++k) {
                            data[i * extent_1 * extent_2 + j * extent_2 + k] =
                              (i + lo[0]) * 4 + (j + lo[1]) * 2 + (k + lo[2]) +
                              1;
                        }
            };

            auto buf = alloc.allocate(fxn, ten_shape);
            REQUIRE(inner_tile_count == 1); // OneBigTile has only 1 tile
            REQUIRE(buf == ten);
        }
    }
}
