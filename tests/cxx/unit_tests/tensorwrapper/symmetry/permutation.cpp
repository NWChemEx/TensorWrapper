#include "../helpers.hpp"
#include <tensorwrapper/symmetry/permutation.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::symmetry;

using mode_index_type = Permutation::mode_index_type;

TEST_CASE("Permutation") {
    Permutation defaulted;
    Permutation one_cycle{0, 1};
    Permutation two_cycles({2, 1, 3}, {4, 5});

    SECTION("Ctors and assignment") {
        SECTION("Default") {
            REQUIRE(defaulted.size() == mode_index_type(0));
            REQUIRE(defaulted.minimum_rank() == mode_index_type(0));
        }

        SECTION("Cycle") {
            REQUIRE(one_cycle.size() == mode_index_type(1));
            REQUIRE(one_cycle.minimum_rank() == mode_index_type(2));

            REQUIRE(two_cycles.size() == mode_index_type(2));
            REQUIRE(two_cycles.minimum_rank() == mode_index_type(6));
        }
    }
}
