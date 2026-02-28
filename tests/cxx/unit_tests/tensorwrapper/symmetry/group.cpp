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
#include <stdexcept>
#include <tensorwrapper/symmetry/group.hpp>
#include <tensorwrapper/symmetry/permutation.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::symmetry;

TEST_CASE("Group") {
    using cycle_type = typename Permutation::cycle_type;
    using label_type = typename Group::label_type;
    using size_type  = typename Group::size_type;

    Permutation p01(4, cycle_type{0, 1});
    Permutation p23(4, cycle_type{2, 3});

    Group empty;
    Group scalar(0);
    Group vector(1);
    Group matrix(2);
    Group tensor(3);
    Group g(p01, p23);

    SECTION("Ctors and assignment") {
        SECTION("Default") {
            REQUIRE(empty.size() == 0);
            REQUIRE(empty.rank() == 0);
        }

        SECTION("Identity") {
            Group i0(0);
            REQUIRE(i0.size() == 0);
            REQUIRE(i0.rank() == 0);

            Group i1(1);
            REQUIRE(i1.size() == 0);
            REQUIRE(i1.rank() == 1);
        }

        SECTION("Value") {
            REQUIRE(g.rank() == 4);
            REQUIRE(g.size() == 2);
            REQUIRE(g.at(0).are_equal(p01));
            REQUIRE(g.at(1).are_equal(p23));

            // Removes duplicates
            Group g2(p01, p01);
            REQUIRE(g2.rank() == 4);
            REQUIRE(g2.size() == 1);
            REQUIRE(g2.at(0).are_equal(p01));

            // Doesn't store identity operations
            Group g3(p01, Permutation{0, 1, 2, 3});
            REQUIRE(g3.rank() == 4);
            REQUIRE(g3.size() == 1);
            REQUIRE(g3.at(0).are_equal(p01));

            // Throws if operations have different ranks
            using error_t = std::runtime_error;
            REQUIRE_THROWS_AS(Group(p01, Permutation(2)), error_t);
        }
        test_copy_move_ctor_and_assignment(empty, g);
    }

    SECTION("count") {
        REQUIRE_FALSE(empty.count(p01));
        REQUIRE_FALSE(scalar.count(p01));
        REQUIRE_FALSE(vector.count(p01));
        REQUIRE_FALSE(matrix.count(p01));
        REQUIRE_FALSE(tensor.count(p01));
        REQUIRE(g.count(p01));
        REQUIRE(g.count(p23));
    }

    SECTION("rank") {
        REQUIRE(empty.rank() == 0);
        REQUIRE(scalar.rank() == 0);
        REQUIRE(vector.rank() == 1);
        REQUIRE(matrix.rank() == 2);
        REQUIRE(tensor.rank() == 3);
        REQUIRE(g.rank() == 4);
    }

    SECTION("slice(initializer_lists)") {
        using except_t = std::runtime_error;
        REQUIRE(empty.slice({}, {}) == empty);
        REQUIRE(scalar.slice({}, {}) == scalar);
        REQUIRE(vector.slice({0}, {1}) == vector);
        REQUIRE(matrix.slice({0, 0}, {1, 1}) == matrix);
        REQUIRE(tensor.slice({0, 0, 0}, {1, 1, 1}) == tensor);

        // NYI for actual symmetries
        REQUIRE_THROWS_AS(g.slice({0, 0, 0, 0}, {1, 1, 1, 1}), except_t);

        // Offset ranks don't match: one empty offset
        REQUIRE_THROWS_AS(vector.slice({}, {1}), except_t);

        // Offset ranks don't match: catch in loop
        REQUIRE_THROWS_AS(matrix.slice({0, 0}, {1}), except_t);

        // Offset ranks don't match: catch after loop
        REQUIRE_THROWS_AS(vector.slice({0}, {1, 1}), except_t);

        // Offset ranks don't match tensor's rank: empty offsets
        REQUIRE_THROWS_AS(vector.slice({}, {}), except_t);

        // Offset ranks don't match tensor's rank: non-empty
        REQUIRE_THROWS_AS(vector.slice({0, 0}, {1, 1}), except_t);

        // first_elem == last_elem
        REQUIRE_THROWS_AS(vector.slice({1}, {1}), except_t);

        // first_elem > last_elem
        REQUIRE_THROWS_AS(vector.slice({1}, {0}), except_t);
    }

    SECTION("slice(containers)") {
        // This overload dispatches to the ranges overload, so we just spot
        // check.

        std::vector<size_type> empty_idx;
        REQUIRE(empty.slice(empty_idx, empty_idx) == empty);
    }

    SECTION("slice(ranges)") {
        // For convenience we thoroughly test this function via the il overload.
        // All overloads dispatch to this one, thus it's thoroughly tested
        // there.

        std::vector<size_type> empty_idx;
        auto eb = empty_idx.begin();
        auto ee = empty_idx.end();
        REQUIRE(empty.slice(eb, ee, eb, ee) == empty);
    }

    SECTION("swap") {
        Group copy_empty(empty);
        Group copy_g(g);

        g.swap(empty);
        REQUIRE(copy_g == empty);
        REQUIRE(copy_empty == g);
    }

    SECTION("operator==") {
        // Default constructed  equals default constructed
        REQUIRE(empty == Group{});

        // Default equal value constructio of scalar identity group
        REQUIRE(empty == Group(0));
        REQUIRE(empty == Group{Permutation(0)});

        // Default does not equal a general value construction
        REQUIRE_FALSE(empty == g);

        // Identity constructed with same rank
        Group g1(1);
        REQUIRE(g1 == Group(1));

        // Identity with different ranks
        REQUIRE_FALSE(g1 == Group(2));

        // Identity with non-identity
        REQUIRE_FALSE(Group(4) == g);

        // Value constructed equal value constructed with same value
        REQUIRE(g == Group{p01, p23});
        REQUIRE(g == Group{p23, p01}); // Order doesn't matter

        // Value constructed with different numbers of elements
        REQUIRE_FALSE(g == Group{p01});

        // Value constructed with different elements
        Permutation p0213{0, 2, 1, 3};
        Permutation p3120{3, 1, 2, 0};
        REQUIRE_FALSE(g == Group{p0213, p3120});
    }

    SECTION("operator!=") {
        // Implemented in terms of operator==, so just spot check
        REQUIRE_FALSE(empty != Group{});
        REQUIRE(empty != g);
    }

    SECTION("at_()") {
        REQUIRE(g.at(0).are_equal(p01));
        REQUIRE(g.at(1).are_equal(p23));
    }

    SECTION("at_() const") {
        REQUIRE(std::as_const(g).at(0).are_equal(p01));
        REQUIRE(std::as_const(g).at(1).are_equal(p23));
    }

    SECTION("size_()") {
        REQUIRE(empty.size() == 0);
        REQUIRE(g.size() == 2);
    }

    SECTION("addition_assignment_") {
        Group empty2;

        SECTION("Identity plus identity") {
            Group g2(2);
            auto g2ij    = g2("i,j");
            auto pempty2 = &(empty2.addition_assignment("i,j", g2ij, g2ij));
            REQUIRE(pempty2 == &empty2);
            REQUIRE(empty2 == g2);
        }

        // Throws if non-trivial symmetry
        using error_t = std::runtime_error;
        label_type ijkl("i,j,k,l");
        auto lg = g("i,j,k,l");
        REQUIRE_THROWS_AS(empty2.addition_assignment(ijkl, lg, lg), error_t);
    }

    SECTION("subtraction_assignment_") {
        Group empty2;

        SECTION("Identity plus identity") {
            Group g2(2);
            auto g2ij    = g2("i,j");
            auto pempty2 = &(empty2.subtraction_assignment("i,j", g2ij, g2ij));
            REQUIRE(pempty2 == &empty2);
            REQUIRE(empty2 == g2);
        }

        // Throws if non-trivial symmetry
        using error_t = std::runtime_error;
        label_type ijkl("i,j,k,l");
        auto lg = g("i,j,k,l");
        REQUIRE_THROWS_AS(empty2.subtraction_assignment(ijkl, lg, lg), error_t);
    }

    SECTION("multiplication_assignment_") {
        Group empty2;

        SECTION("Identity plus identity") {
            Group g2(2);
            auto g2ij = g2("i,j");
            auto pempty2 =
              &(empty2.multiplication_assignment("i,j", g2ij, g2ij));
            REQUIRE(pempty2 == &empty2);
            REQUIRE(empty2 == g2);
        }

        // Throws if non-trivial symmetry
        using error_t = std::runtime_error;
        label_type ijkl("i,j,k,l");
        auto lg = g("i,j,k,l");
        REQUIRE_THROWS_AS(empty2.multiplication_assignment(ijkl, lg, lg),
                          error_t);
    }

    SECTION("permute_assignment_") {
        Group empty2;

        SECTION("Permute identity") {
            Group g2(2);
            auto g2ij    = g2("i,j");
            auto pempty2 = &(empty2.permute_assignment("i,j", g2ij));
            REQUIRE(pempty2 == &empty2);
            REQUIRE(empty2 == g2);
        }

        // Throws if non-trivial symmetry
        using error_t = std::runtime_error;
        label_type ijkl("i,j,k,l");
        auto lg = g("i,j,k,l");
        REQUIRE_THROWS_AS(empty2.permute_assignment(ijkl, lg), error_t);
    }
}
