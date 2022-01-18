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
    using tensor_type = TA::TSpArrayD;
    auto palloc       = tensorwrapper::tensor::default_allocator<field_type>();

    using allocator_type = typename decltype(palloc)::element_type;
    using extents_type   = typename allocator_type::extents_type;
    using shape_type     = typename allocator_type::shape_type;

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

    SECTION("Comparisons") {
        const auto prhs =
          tensorwrapper::tensor::default_allocator<field_type>();
        REQUIRE(*palloc == *prhs);
        REQUIRE_FALSE(*palloc != *prhs);

        REQUIRE(palloc->is_equal(*prhs));
    }
}
