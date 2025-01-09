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
#include <tensorwrapper/tensorwrapper.hpp>

using namespace tensorwrapper;

TEMPLATE_LIST_TEST_CASE("DSL", "", testing::dsl_types) {
    using object_type = TestType;

    auto scalar_values = testing::scalar_values();
    auto vector_values = testing::vector_values();
    auto matrix_values = testing::matrix_values();

    auto value0 = std::get<object_type>(scalar_values);
    auto value1 = std::get<object_type>(vector_values);
    auto value2 = std::get<object_type>(matrix_values);

    SECTION("assignment") {
        value0("i,j") = value2("i,j");
        REQUIRE(value0 == value2);
    }

    SECTION("permutation") {
        value0("j,i") = value2("i,j");

        value1.permute_assignment("i,j", value2("j,i"));
        REQUIRE(value1.are_equal(value0));
    }

    SECTION("addition") {
        value0("i,j") = value2("i,j") + value2("i,j");

        value1.addition_assignment("i,j", value2("i,j"), value2("i,j"));
        REQUIRE(value1.are_equal(value0));
    }

    SECTION("subtraction") {
        value0("i,j") = value2("i,j") - value2("i,j");

        value1.subtraction_assignment("i,j", value2("i,j"), value2("i,j"));
        REQUIRE(value1.are_equal(value0));
    }

    SECTION("multiplication") {
        value0("i,j") = value2("i,j") * value2("i,j");

        value1.multiplication_assignment("i,j", value2("i,j"), value2("i,j"));
        REQUIRE(value1.are_equal(value0));
    }
}