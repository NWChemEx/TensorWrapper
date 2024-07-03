#include "../helpers.hpp"
#include <tensorwrapper/symmetry/group.hpp>
#include <tensorwrapper/symmetry/permutation.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::symmetry;

TEST_CASE("Group") {
    Permutation p01{0, 1};
    Permutation p123{1, 2, 3};

    Group empty;
    Group g(p01, p123);

    SECTION("Ctors and assignment") {
        SECTION("Default") { REQUIRE(empty.size() == 0); }

        SECTION("Value") {
            REQUIRE(g.size() == 2);
            REQUIRE(g.at(0).are_equal(p01));
            REQUIRE(g.at(1).are_equal(p123));

            // Removes duplicates
            Group g2(p01, p01);
            REQUIRE(g2.size() == 1);
            REQUIRE(g2.at(0).are_equal(p01));

            // Doesn't store identity operations
            Group g3(p01, Permutation{}, Permutation{2});
            REQUIRE(g3.size() == 1);
            REQUIRE(g3.at(0).are_equal(p01));
        }
        test_copy_move_ctor_and_assignment(empty, g);
    }

    SECTION("count") {
        REQUIRE_FALSE(empty.count(p01));
        REQUIRE(g.count(p01));
        REQUIRE(g.count(p123));
    }

    SECTION("swap") {
        Group copy_empty(empty);
        Group copy_g(g);

        g.swap(empty);
        REQUIRE(copy_g == empty);
        REQUIRE(copy_empty == g);
    }

    SECTION("operator==") {
        REQUIRE(empty == Group{});
        REQUIRE_FALSE(empty == g);

        REQUIRE(g == Group{p01, p123});
        REQUIRE(g == Group{p123, p01}); // Order doesn't matter

        REQUIRE_FALSE(g == Group{p01});
        REQUIRE_FALSE(g == Group{p01, p123, Permutation{4, 5}});
    }

    SECTION("operator!=") {
        // Implemented in terms of operator==, so just spot check
        REQUIRE_FALSE(empty != Group{});
        REQUIRE(empty != g);
    }

    SECTION("at_()") {
        REQUIRE(g.at(0).are_equal(p01));
        REQUIRE(g.at(1).are_equal(p123));
    }

    SECTION("at_() const") {
        REQUIRE(std::as_const(g).at(0).are_equal(p01));
        REQUIRE(std::as_const(g).at(1).are_equal(p123));
    }

    SECTION("size_()") {
        REQUIRE(empty.size() == 0);
        REQUIRE(g.size() == 2);
    }
}
