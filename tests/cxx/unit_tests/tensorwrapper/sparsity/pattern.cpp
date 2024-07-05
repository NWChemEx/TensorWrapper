#include "../helpers.hpp"
#include <tensorwrapper/sparsity/pattern.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::sparsity;

TEST_CASE("Pattern") {
    Pattern defaulted;

    SECTION("Ctors, assignment") {
        SECTION("Default") {}

        test_copy_move_ctor_and_assignment(defaulted);
    }

    SECTION("operator==") { REQUIRE(defaulted == Pattern{}); }

    SECTION("operator!=") {
        // Just spot check because it is implemented in terms of operator==
        REQUIRE_FALSE(defaulted != Pattern{});
    }
}
