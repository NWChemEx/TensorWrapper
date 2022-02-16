#if 0
#include "tensorwrapper/tensor/allocators/allocators.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;

/* Testing strategy:
 *
 * SingleElementTiles has no state. So there's not much to test aside from
 * ensuring clone makes a polymorphic copy (verified by calling
 * make_tiled_range) and that make_tiled_range works as expected.
 *
 */
TEST_CASE("SingleElementTiles") {
    using field_type   = tensorwrapper::tensor::field::Scalar;
    using alloc_type   = SingleElementTiles<field_type>;
    using extents_type = typename alloc_type::extents_type;
    using tr_type      = typename alloc_type::tiled_range_type;

    alloc_type a;

    extents_type shape{3, 4, 5};
    tr_type corr{{0, 1, 2, 3}, {0, 1, 2, 3, 4}, {0, 1, 2, 3, 4, 5}};

    SECTION("make_tiled_range") { REQUIRE(a.make_tiled_range(shape) == corr); }

    SECTION("clone") {
        auto pb = a.clone();
        REQUIRE(pb->make_tiled_range(shape) == corr);
    }

    SECTION("is_equal") {
        SECTION("Both SingleElementTiles") {
            alloc_type b;
            REQUIRE(a.is_equal(b));
            REQUIRE(b.is_equal(a));
        }

        SECTION("Different derived types") {
            OneBigTile<field_type> c;
            REQUIRE_FALSE(c.is_equal(a));
            REQUIRE_FALSE(a.is_equal(c));
        }
    }
}
#endif
