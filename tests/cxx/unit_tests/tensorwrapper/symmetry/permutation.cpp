/*
 * Copyright 2024 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../helpers.hpp"
#include <tensorwrapper/symmetry/permutation.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::symmetry;

using mode_index_type = Permutation::mode_index_type;
using cycle_type      = Permutation::cycle_type;

TEST_CASE("Permutation") {
    Permutation defaulted;
    Permutation one_cycle{0, 1};
    Permutation two_cycles(cycle_type{2, 1, 3}, cycle_type{4, 5});

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
            REQUIRE(two_cycles.at(0) == c132); // Canonicalization must work
            REQUIRE(two_cycles.at(1) == c45);

            SECTION("Removes trivial cycles") {
                Permutation one_trivial_cycle{0};
                REQUIRE(one_trivial_cycle.size() == 0);
                REQUIRE(one_trivial_cycle.minimum_rank() == 0);

                Permutation two_trivial_cycles(cycle_type{0}, cycle_type{1});
                REQUIRE(two_trivial_cycles.size() == 0);
                REQUIRE(two_trivial_cycles.minimum_rank() == 0);

                Permutation one_trivial_one_real(cycle_type{4},
                                                 cycle_type{0, 1});
                REQUIRE(one_trivial_one_real.size() == 1);
                REQUIRE(one_trivial_one_real.minimum_rank() == 2);
            }

            using except = std::runtime_error;

            SECTION("Error if invalid cycle") {
                REQUIRE_THROWS_AS((Permutation{0, 0}), except);
            }

            SECTION("Error if cycles overlap") {
                REQUIRE_THROWS_AS(
                  (Permutation(cycle_type{0, 1}, cycle_type{1, 2})), except);
            }

            test_copy_move_ctor_and_assignment(defaulted, one_cycle,
                                               two_cycles);
        }
    }

    SECTION("minimum_rank") {
        REQUIRE(defaulted.minimum_rank() == 0);
        REQUIRE(one_cycle.minimum_rank() == 2);
        REQUIRE(two_cycles.minimum_rank() == 6);
    }

    SECTION("operator[]") {
        REQUIRE(one_cycle[0] == c01);
        REQUIRE(two_cycles[0] == c132);
        REQUIRE(two_cycles[1] == c45);
    }

    SECTION("at") {
        REQUIRE_THROWS_AS(defaulted.at(0), std::out_of_range);

        REQUIRE(one_cycle.at(0) == c01);
        REQUIRE_THROWS_AS(one_cycle.at(1), std::out_of_range);

        REQUIRE(two_cycles.at(0) == c132);
        REQUIRE(two_cycles.at(1) == c45);
        REQUIRE_THROWS_AS(two_cycles.at(2), std::out_of_range);
    }

    SECTION("size") {
        REQUIRE(defaulted.size() == 0);
        REQUIRE(one_cycle.size() == 1);
        REQUIRE(two_cycles.size() == 2);
    }

    SECTION("swap") {
        Permutation copy_defaulted(defaulted);
        Permutation copy_one_cycle(one_cycle);

        one_cycle.swap(defaulted);

        REQUIRE(one_cycle == copy_defaulted);
        REQUIRE(defaulted == copy_one_cycle);
    }

    SECTION("operator==") {
        // Defaulted equals another defaulted object
        REQUIRE(defaulted == Permutation{});

        // Defaulted equals an object with only trivial cycles
        REQUIRE(defaulted == Permutation{1});
        REQUIRE(defaulted == Permutation(cycle_type{0}, cycle_type{1}));

        // Defaulted does not equal an object with non-trivial cycles
        REQUIRE_FALSE(defaulted == one_cycle);

        // Values input in same order
        REQUIRE(two_cycles ==
                Permutation(cycle_type{2, 1, 3}, cycle_type{4, 5}));

        // Values input in different order
        REQUIRE(two_cycles ==
                Permutation(cycle_type{4, 5}, cycle_type{1, 3, 2}));

        // Different number of cycles
        REQUIRE_FALSE(one_cycle == two_cycles);

        // Different cycles
        REQUIRE_FALSE(two_cycles ==
                      Permutation(cycle_type{1, 2}, cycle_type{3, 4, 5}));
    }

    SECTION("operator!=") {
        // Implemented in terms of operator==, just spot check
        REQUIRE_FALSE(defaulted != Permutation{});
        REQUIRE(one_cycle != two_cycles);
    }

    SECTION("virtual methods") {
        using const_base_reference = Permutation::const_base_reference;

        SECTION("clone") {
            const_base_reference as_base = two_cycles;
            auto pcopy_two_cycles        = as_base.clone();
            REQUIRE(pcopy_two_cycles->are_equal(as_base));
        }

        SECTION("is_identity") {
            const_base_reference defaulted_base = defaulted;
            REQUIRE(defaulted.is_identity());
            REQUIRE(defaulted_base.is_identity());

            const_base_reference one_cycle_base = one_cycle;
            REQUIRE_FALSE(one_cycle.is_identity());
            REQUIRE_FALSE(one_cycle_base.is_identity());

            const_base_reference two_cycles_base = two_cycles;
            REQUIRE_FALSE(two_cycles.is_identity());
            REQUIRE_FALSE(two_cycles_base.is_identity());
        }

        SECTION("are_equal") {
            const_base_reference one_base = one_cycle;
            const_base_reference two_base = two_cycles;
            REQUIRE_FALSE(one_base.are_equal(two_base));
            REQUIRE(Permutation{0, 1}.are_equal(one_base));
        }
    }
}
