/*
 * Copyright 2025 NWChemEx-Project
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