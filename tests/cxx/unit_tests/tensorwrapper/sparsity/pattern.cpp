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
#include "../testing/testing.hpp"
#include <tensorwrapper/sparsity/pattern.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::sparsity;

TEST_CASE("Pattern") {
    using size_type = Pattern::size_type;

    Pattern defaulted;
    Pattern p0(0);
    Pattern p1(1);
    Pattern p2(2);
    Pattern p3(3);

    SECTION("Ctors, assignment") {
        SECTION("Default") { REQUIRE(defaulted.rank() == 0); }

        SECTION("rank ctor") {
            REQUIRE(Pattern(0).rank() == 0);
            REQUIRE(p1.rank() == 1);
            REQUIRE(Pattern(2).rank() == 2);
        }

        test_copy_move_ctor_and_assignment(defaulted, p1);
    }

    SECTION("rank") {
        REQUIRE(defaulted.rank() == 0);
        REQUIRE(p0.rank() == 0);
        REQUIRE(p1.rank() == 1);
        REQUIRE(p2.rank() == 2);
        REQUIRE(p3.rank() == 3);
    }

    SECTION("slice(initializer_lists)") {
        using except_t = std::runtime_error;
        REQUIRE(defaulted.slice({}, {}) == defaulted);
        REQUIRE(p0.slice({}, {}) == p0);
        REQUIRE(p1.slice({0}, {1}) == p1);
        REQUIRE(p2.slice({0, 0}, {1, 1}) == p2);
        REQUIRE(p3.slice({0, 0, 0}, {1, 1, 1}) == p3);

        // Offset ranks don't match: one empty offset
        REQUIRE_THROWS_AS(p1.slice({}, {1}), except_t);

        // Offset ranks don't match: catch in loop
        REQUIRE_THROWS_AS(p2.slice({0, 0}, {1}), except_t);

        // Offset ranks don't match: catch after loop
        REQUIRE_THROWS_AS(p1.slice({0}, {1, 1}), except_t);

        // Offset ranks don't match tensor's rank: empty offsets
        REQUIRE_THROWS_AS(p1.slice({}, {}), except_t);

        // Offset ranks don't match tensor's rank: non-empty
        REQUIRE_THROWS_AS(p1.slice({0, 0}, {1, 1}), except_t);

        // first_elem == last_elem
        REQUIRE_THROWS_AS(p1.slice({1}, {1}), except_t);

        // first_elem > last_elem
        REQUIRE_THROWS_AS(p1.slice({1}, {0}), except_t);
    }

    SECTION("slice(containers)") {
        // This overload dispatches to the ranges overload, so we just spot
        // check.

        std::vector<size_type> empty_idx;
        REQUIRE(defaulted.slice(empty_idx, empty_idx) == defaulted);
    }

    SECTION("slice(ranges)") {
        // For convenience we thoroughly test this function via the il overload.
        // All overloads dispatch to this one, thus it's thoroughly tested
        // there.

        std::vector<size_type> empty_idx;
        auto eb = empty_idx.begin();
        auto ee = empty_idx.end();
        REQUIRE(defaulted.slice(eb, ee, eb, ee) == defaulted);
    }

    SECTION("operator==") {
        // Defaulted is same as another defaulted
        REQUIRE(defaulted == Pattern{});

        // Defaulted is same as scalar
        REQUIRE(defaulted == Pattern(0));

        // Defaulted is not same as vector
        REQUIRE_FALSE(defaulted == p1);

        // Vector equals vector
        REQUIRE(p1 == Pattern(1));

        // Vector not same as matrix
        REQUIRE_FALSE(p1 == Pattern(2));
    }

    SECTION("operator!=") {
        // Just spot check because it is implemented in terms of operator==
        REQUIRE_FALSE(defaulted != Pattern{});
        REQUIRE(defaulted != p1);
    }

    SECTION("clone") {
        auto pdefaulted = defaulted.clone();
        REQUIRE(pdefaulted->are_equal(defaulted));

        auto pp1 = p1.clone();
        REQUIRE(pp1->are_equal(p1));
    }

    SECTION("are_equal") {
        // Just calls operator== so spot check.
        REQUIRE(defaulted.are_equal(Pattern{}));
        REQUIRE_FALSE(defaulted.are_equal(p1));
    }

    SECTION("addition_assignment") {
        Pattern rv;
        auto prv = &(rv.addition_assignment("i", p1("i"), p1("i")));
        REQUIRE(prv == &rv);
        REQUIRE(rv == p1);
    }

    SECTION("subtraction_assignment") {
        Pattern rv;
        auto prv = &(rv.subtraction_assignment("i", p1("i"), p1("i")));
        REQUIRE(prv == &rv);
        REQUIRE(rv == p1);
    }

    SECTION("multiplication_assignment") {
        Pattern rv;
        auto prv = &(rv.multiplication_assignment("i", p1("i"), p1("i")));
        REQUIRE(prv == &rv);
        REQUIRE(rv == p1);
    }

    SECTION("permute_assignment") {
        Pattern rv;
        auto prv = &(rv.permute_assignment("i", p1("i")));
        REQUIRE(prv == &rv);
        REQUIRE(rv == p1);
    }
}
