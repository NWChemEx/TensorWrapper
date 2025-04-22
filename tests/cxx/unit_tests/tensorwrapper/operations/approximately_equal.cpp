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

/* Notes on testing.
 *
 * - Because of how floating point conversions work, a difference of the
 *   tolerance may be equal, slightly less than, or slightly more than the
 *   tolerance converted to a different floating point type. We do not test for
 *   exact equality to the tolerance.
 * - We can test for positive and negative differences by flipping the order of
 *   arguments.
 */

TEMPLATE_LIST_TEST_CASE("approximately_equal", "",
                        types::floating_point_types) {
    auto pscalar = testing::eigen_scalar<TestType>();
    pscalar->set_data(0, 42.0);
    auto pvector = testing::eigen_vector<TestType>(2);
    pvector->set_data(0, 1.23);
    pvector->set_data(1, 2.34);

    auto pscalar2 = testing::eigen_scalar<TestType>();
    pscalar2->set_data(0, 42.0);
    auto pvector2 = testing::eigen_vector<TestType>(2);
    pvector2->set_data(0, 1.23);
    pvector2->set_data(1, 2.34);

    shape::Smooth s0{};
    shape::Smooth s1{2};

    Tensor scalar(s0, std::move(pscalar));
    Tensor vector(s1, std::move(pvector));

    SECTION("different ranks") {
        REQUIRE_FALSE(approximately_equal(scalar, vector));
        REQUIRE_FALSE(approximately_equal(vector, scalar));
    }

    SECTION("Same values") {
        Tensor scalar2(s0, std::move(pscalar2));
        Tensor vector2(s1, std::move(pvector2));

        REQUIRE(approximately_equal(scalar, scalar2));
        REQUIRE(approximately_equal(scalar2, scalar));
        REQUIRE(approximately_equal(vector, vector2));
        REQUIRE(approximately_equal(vector2, vector));
    }

    SECTION("Differ by more than default tolerance") {
        double value = 1e-1;
        pscalar2->set_data(0, 42.0 + value);
        pvector2->set_data(0, 1.23 + value);
        Tensor scalar2(s0, std::move(pscalar2));
        Tensor vector2(s1, std::move(pvector2));
        REQUIRE_FALSE(approximately_equal(scalar, scalar2));
        REQUIRE_FALSE(approximately_equal(scalar2, scalar));
        REQUIRE_FALSE(approximately_equal(vector, vector2));
        REQUIRE_FALSE(approximately_equal(vector2, vector));
    }

    SECTION("Differ by less than default tolerance") {
        double value = 1e-17;
        pscalar2->set_data(0, 42.0 + value);
        pvector2->set_data(0, 1.23 + value);
        Tensor scalar2(s0, std::move(pscalar2));
        Tensor vector2(s1, std::move(pvector2));
        REQUIRE(approximately_equal(scalar, scalar2));
        REQUIRE(approximately_equal(scalar2, scalar));
        REQUIRE(approximately_equal(vector, vector2));
        REQUIRE(approximately_equal(vector2, vector));
    }

    SECTION("Differ by more than provided tolerance") {
        float value = 1e-1;
        pscalar2->set_data(0, 43.0);
        pvector2->set_data(0, 2.23);
        Tensor scalar2(s0, std::move(pscalar2));
        Tensor vector2(s1, std::move(pvector2));
        REQUIRE_FALSE(approximately_equal(scalar, scalar2, value));
        REQUIRE_FALSE(approximately_equal(scalar2, scalar, value));
        REQUIRE_FALSE(approximately_equal(vector, vector2, value));
        REQUIRE_FALSE(approximately_equal(vector2, vector, value));
    }

    SECTION("Differ by less than provided tolerance") {
        double value = 1e-10;
        pscalar2->set_data(0, 42.0 + value);
        pvector2->set_data(0, 1.23 + value);
        Tensor scalar2(s0, std::move(pscalar2));
        Tensor vector2(s1, std::move(pvector2));
        REQUIRE(approximately_equal(scalar, scalar2, 1e-1));
        REQUIRE(approximately_equal(scalar2, scalar, 1e-1));
        REQUIRE(approximately_equal(vector, vector2, 1e-1));
        REQUIRE(approximately_equal(vector2, vector, 1e-1));
    }
}
