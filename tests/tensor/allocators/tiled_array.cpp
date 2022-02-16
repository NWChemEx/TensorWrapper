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
