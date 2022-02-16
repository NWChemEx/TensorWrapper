#include "../buffer/make_pimpl.hpp"
#include "tensorwrapper/ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/allocators/allocators.hpp"
#include <catch2/catch.hpp>

/* Testing Strategy:
 *
 * The unit tests focuse on the parts of the Allocator hierarchy which are
 * implemented in the Allocator class (as opposed to the derived classes).
 * Unit tests for the derived classes focus on the parts they implement.
 */

TEST_CASE("Allocator") {
    using field_type  = tensorwrapper::tensor::field::Scalar;
    using buffer_type = tensorwrapper::tensor::buffer::Buffer<field_type>;
    auto palloc       = tensorwrapper::tensor::default_allocator<field_type>();

    using allocator_type = typename decltype(palloc)::element_type;
    using extents_type   = typename allocator_type::extents_type;
    using shape_type     = typename allocator_type::shape_type;

#if 0
    // Assumes allocator uses this world too
    auto& world = TA::get_default_world();

    SECTION("new_tensor(shape)") {
        shape_type shape0(extents_type{2});

        auto t = palloc->new_tensor(shape0);

        // b/c the tensor isn't initialized we can't directly compare them, so
        // instead we compare the tiled ranges

        auto corr = palloc->make_tiled_range(shape0.extents());
        REQUIRE(std::get<0>(t).trange() == corr);
        // TODO: proper unit test when runtime is comparable
        REQUIRE_NOTHROW(palloc->runtime());
    }

    SECTION("new_tensor(vector)") {
        auto t  = palloc->new_tensor({1.0, 2.0, 3.0});
        auto tr = palloc->make_tiled_range(extents_type{3});
        tensor_type corr(world, tr, {1.0, 2.0, 3.0});
        REQUIRE(corr == std::get<0>(t));
    }

    SECTION("new_tensor(matrix)") {
        auto t  = palloc->new_tensor({{1.0, 2.0}, {3.0, 4.0}});
        auto tr = palloc->make_tiled_range(extents_type{2, 2});
        tensor_type corr(world, tr, {{1.0, 2.0}, {3.0, 4.0}});
        REQUIRE(corr == std::get<0>(t));
    }

    SECTION("new_tensor(rank 3 tensor)") {
        auto t = palloc->new_tensor(
          {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
        auto tr = palloc->make_tiled_range(extents_type{2, 2, 2});
        tensor_type corr(world, tr,
                         {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}});
        REQUIRE(corr == std::get<0>(t));
    }

    SECTION("new_tensor(rank 4 tensor)") {
        auto t = palloc->new_tensor(
          {{{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}},
           {{{9.0, 10.0}, {11.0, 12.0}}, {{13.0, 14.0}, {15.0, 16.0}}}});
        auto tr = palloc->make_tiled_range(extents_type{2, 2, 2, 2});
        tensor_type corr(
          world, tr,
          {{{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}},
           {{{9.0, 10.0}, {11.0, 12.0}}, {{13.0, 14.0}, {15.0, 16.0}}}});
        REQUIRE(corr == std::get<0>(t));
    }
#endif

    SECTION("Comparisons") {
        const auto prhs =
          tensorwrapper::tensor::default_allocator<field_type>();
        REQUIRE(*palloc == *prhs);
        REQUIRE_FALSE(*palloc != *prhs);

        REQUIRE(palloc->is_equal(*prhs));
    }

    SECTION("Clone") {
        const auto copy = palloc->clone();
        REQUIRE(*palloc == *copy);
        REQUIRE_FALSE(*palloc != *copy);
        REQUIRE(palloc->is_equal(*copy));
    }

    auto&& [pvec, pmat, pt3d] = testing::make_pimpl<field_type>();
    SECTION("allocate(rank 1 tensor)") {
        buffer_type vec(pvec->clone());

        auto fxn = [](std::vector<size_t> lo, std::vector<size_t> up,
                      double* data) {
            REQUIRE(lo.size() == 1);
            REQUIRE(up.size() == 1);
            REQUIRE(lo[0] >= 0);
            REQUIRE(up[0] <= 3);
            REQUIRE(lo[0] < up[0]);
            size_t extent = up[0] - lo[0];
            for(auto i = 0; i < extent; ++i) { data[i] = i + lo[0] + 1; }
        };

        extents_type vec_extents{3};
        shape_type vec_shape(vec_extents);
        auto buf = palloc->allocate(fxn, vec_shape);
        REQUIRE(buf == vec);
    }

    SECTION("allocate(rank 2 tensor)") {
        buffer_type mat(pmat->clone());

        auto fxn = [](std::vector<size_t> lo, std::vector<size_t> up,
                      double* data) {
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
                    data[i * extent_1 + j] = (i + lo[0]) * 2 + (j + lo[1]) + 1;
                }
        };

        extents_type mat_extents{2, 2};
        shape_type mat_shape(mat_extents);
        auto buf = palloc->allocate(fxn, mat_shape);
        REQUIRE(buf == mat);
    }

    SECTION("allocate(rank 3 tensor)") {
        buffer_type ten3(pt3d->clone());

        auto fxn = [](std::vector<size_t> lo, std::vector<size_t> up,
                      double* data) {
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
                          (i + lo[0]) * 4 + (j + lo[1]) * 2 + (k + lo[2]) + 1;
                    }
        };

        extents_type ten_extents{2, 2, 2};
        shape_type ten_shape(ten_extents);
        auto buf = palloc->allocate(fxn, ten_shape);
        REQUIRE(buf == ten3);
    }
}
