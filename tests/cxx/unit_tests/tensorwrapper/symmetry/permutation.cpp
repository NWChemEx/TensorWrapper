#include "../helpers.hpp"
#include <tensorwrapper/symmetry/permutation.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::symmetry;

using mode_index_type = Permutation::mode_index_type;
using cycle_type      = Permutation::cycle_type;

TEST_CASE("Permutation") {
    Permutation defaulted;
    Permutation one_cycle{0, 1};
    Permutation two_cycles{{2, 1, 3}, {4, 5}};

    cycle_type c01{0, 1};
    cycle_type c132{1, 3, 2};
    cycle_type c45{4, 5};

    SECTION("Ctors and assignment") {
        SECTION("Default") {
            REQUIRE(defaulted.size() == mode_index_type(0));
            REQUIRE(defaulted.minimum_rank() == mode_index_type(0));
        }

        SECTION("Cycle") {
            REQUIRE(one_cycle.size() == mode_index_type(1));
            REQUIRE(one_cycle.minimum_rank() == mode_index_type(2));
            REQUIRE(one_cycle.at(0) == c01);

            REQUIRE(two_cycles.size() == mode_index_type(2));
            REQUIRE(two_cycles.minimum_rank() == mode_index_type(6));
            REQUIRE(two_cycles.at(0) ==
                    c132); // Assumes canonicalization worked
            REQUIRE(two_cycles.at(1) == c45);

            SECTION("Removes trivial cycles") {
                Permutation one_trivial_cycle{0};
                REQUIRE(one_trivial_cycle.size() == 0);
                REQUIRE(one_trivial_cycle.minimum_rank() == 0);

                Permutation two_trivial_cycles{{0}, {1}};
                REQUIRE(two_trivial_cycles.size() == 0);
                REQUIRE(two_trivial_cycles.minimum_rank() == 0);

                Permutation one_trivial_one_real{{4}, {0, 1}};
                REQUIRE(one_trivial_one_real.size() == 1);
                REQUIRE(one_trivial_one_real.minimum_rank() == 2);
            }

            using except = std::runtime_error;

            SECTION("Error if invalid cycle") {
                REQUIRE_THROWS_AS((Permutation{0, 0}), except);
            }

            SECTION("Error if cycles overlap") {
                REQUIRE_THROWS_AS((Permutation{{0, 1}, {1, 2}}), except);
            }
        }
    }
}
