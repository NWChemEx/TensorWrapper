#include "../testing/testing.hpp"

using namespace tensorwrapper;
using namespace operations;

TEST_CASE("approximately_equal") {
    Tensor scalar(42.0);
    Tensor vector{1.23, 2.34};

    SECTION("different ranks") {
        REQUIRE_FALSE(approximately_equal(scalar, vector));
    }

    SECTION("Same values") {
        REQUIRE(approximately_equal(scalar, Tensor(42.0)));
        REQUIRE(approximately_equal(vector, Tensor{1.23, 2.34}));
    }

    SECTION("Differ by default tolerance") {
        double value = 1e-16;
        REQUIRE_FALSE(approximately_equal(Tensor(0.0), Tensor(value)));
        REQUIRE_FALSE(
          approximately_equal(Tensor{1.23, 0.0}, Tensor{1.23, value}));
        REQUIRE_FALSE(
          approximately_equal(Tensor{0.0, 2.34}, Tensor{value, 2.34}));
    }

    SECTION("Differ by more than default tolerance") {
        double value = 1e-16;
        REQUIRE_FALSE(approximately_equal(scalar, Tensor(value)));
        REQUIRE_FALSE(approximately_equal(vector, Tensor{1.23, value}));
        REQUIRE_FALSE(approximately_equal(vector, Tensor{value, 2.34}));
    }

    SECTION("Differ by less than default tolerance") {
        double value = 1e-17;
        REQUIRE(approximately_equal(Tensor(0.0), Tensor(value)));
        REQUIRE(approximately_equal(Tensor{1.23, 0.0}, Tensor{1.23, value}));
        REQUIRE(approximately_equal(Tensor{0.0, 2.34}, Tensor{value, 2.34}));
    }

    SECTION("Differ by provided tolerance") {
        double value = 1e-1;
        REQUIRE_FALSE(approximately_equal(Tensor(0.0), Tensor(value), value));
        REQUIRE_FALSE(
          approximately_equal(Tensor{1.23, 0.0}, Tensor{1.23, value}, value));
        REQUIRE_FALSE(
          approximately_equal(Tensor{0.0, 2.34}, Tensor{value, 2.34}, value));
    }

    SECTION("Differ by more than provided tolerance") {
        double value = 1e-1;
        REQUIRE_FALSE(approximately_equal(scalar, Tensor(value), value));
        REQUIRE_FALSE(approximately_equal(vector, Tensor{1.23, value}, value));
        REQUIRE_FALSE(approximately_equal(vector, Tensor{value, 2.34}, value));
    }

    SECTION("Differ by less than provided tolerance") {
        double value = 1e-2;
        REQUIRE(approximately_equal(Tensor(0.0), Tensor(value), 1e-1));
        REQUIRE(
          approximately_equal(Tensor{1.23, 0.0}, Tensor{1.23, value}, 1e-1));
        REQUIRE(
          approximately_equal(Tensor{0.0, 2.34}, Tensor{value, 2.34}, 1e-1));
    }
}