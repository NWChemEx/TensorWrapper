#include "../helpers.hpp"
#include <tensorwrapper/detail_/view_traits.hpp>

using namespace tensorwrapper::detail_;

TEST_CASE("is_mutable_to_immutable_cast_v") {
    // N.B. Only the const-ness of the types and whether they differ by
    // const-ness should matter
    STATIC_REQUIRE(is_mutable_to_immutable_cast_v<double, const double>);

    STATIC_REQUIRE_FALSE(is_mutable_to_immutable_cast_v<int, const double>);
    STATIC_REQUIRE_FALSE(is_mutable_to_immutable_cast_v<const double, double>);
    STATIC_REQUIRE_FALSE(is_mutable_to_immutable_cast_v<double, double>);
    STATIC_REQUIRE_FALSE(
      is_mutable_to_immutable_cast_v<const double, const double>);
}

TEST_CASE("enable_if_mutable_to_immutable_cast_t") {
    STATIC_REQUIRE(
      std::is_same_v<
        enable_if_mutable_to_immutable_cast_t<double, const double>, void>);
}