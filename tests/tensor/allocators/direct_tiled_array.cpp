#include "../buffer/make_pimpl.hpp"
#include "../shapes/make_tot_shape.hpp"
#include "tensorwrapper/tensor/allocator/allocator.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;

TEMPLATE_TEST_CASE("DirectTiledArrayAllocator Generic", "[allocator][ta]",
                   field::Scalar, field::Tensor) {
    using field_type = TestType;
    using alloc_type = allocator::DirectTiledArrayAllocator<field_type>;

    alloc_type defaulted;
    alloc_type non_default("test");

    SECTION("Default State") { REQUIRE(defaulted.fxn_id() == ""); }

    SECTION("Manual Ctor") {
        REQUIRE(non_default.fxn_id() == "test");

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

TEST_CASE("DirectTiledArrayAllocator<Scalar>") {
    using field_type     = field::Scalar;
    using buffer_type    = buffer::Buffer<field_type>;
    using allocator_type = allocator::DirectTiledArrayAllocator<field_type>;

    using ta_trange_type = TA::TiledRange;

    using allocator::ta::Distribution;
    using allocator::ta::Storage;
    using allocator::ta::Tiling;

    using extents_type = typename allocator_type::extents_type;
    using shape_type   = typename allocator_type::shape_type;

    auto&& [pvec, pmat, pt3d] = testing::make_direct_pimpl();

    extents_type vec_extents{3};
    extents_type mat_extents{2, 2};
    extents_type ten_extents{2, 2, 2};

    shape_type vec_shape(vec_extents);
    shape_type mat_shape(mat_extents);
    shape_type ten_shape(ten_extents);

    SECTION("OneBigTile") {
        // Default tiling is OneBigTile
        buffer_type vec(pvec->clone());
        buffer_type mat(pmat->clone());
        buffer_type ten(pt3d->clone());

        allocator_type alloc1("tile"), alloc2("scalar");

        auto tile_fxn = [&](std::vector<size_t> lo, std::vector<size_t> up,
                            double* data) {
            if(lo.size() == 1) {
                for(auto i = 0; i < 3; ++i) { data[i] = i + 1.0; }
            } else if(lo.size() == 2) {
                for(auto i = 0; i < 4; ++i) { data[i] = i + 1.0; }
            } else {
                for(auto i = 0; i < 8; ++i) { data[i] = i + 1.0; }
            }
        };

        auto scalar_fxn = [&](std::vector<size_t> idx) -> double {
            auto n_dims  = idx.size();
            double value = 1.0;
            for(auto i = 0; i < n_dims; ++i) {
                value += std::pow(2.0, i) * idx[n_dims - 1 - i];
            }
            return value;
        };

        SECTION("allocate(rank 1) - tile op") {
            auto buf = alloc1.allocate(tile_fxn, vec_shape);
            REQUIRE(*buf == vec);
        }

        SECTION("allocate(rank 1) - scalar op") {
            auto buf = alloc2.allocate(scalar_fxn, vec_shape);
            REQUIRE(*buf == vec);
        }

        SECTION("allocate(rank 2) - tile op") {
            auto buf = alloc1.allocate(tile_fxn, mat_shape);
            REQUIRE(*buf == mat);
        }

        SECTION("allocate(rank 2) - scalar op") {
            auto buf = alloc2.allocate(scalar_fxn, mat_shape);
            REQUIRE(*buf == mat);
        }

        SECTION("allocate(rank 3) - tile op") {
            auto buf = alloc1.allocate(tile_fxn, ten_shape);
            REQUIRE(*buf == ten);
        }

        SECTION("allocate(rank 3) - scalar op") {
            auto buf = alloc2.allocate(scalar_fxn, ten_shape);
            REQUIRE(*buf == ten);
        }
    }
}
