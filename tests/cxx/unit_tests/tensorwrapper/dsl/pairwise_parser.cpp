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

using namespace tensorwrapper;

using test_types = std::tuple<shape::Smooth>;

TEMPLATE_LIST_TEST_CASE("PairwiseParser", "", test_types) {
    using object_type = TestType;
    using base_type   = typename object_type::dsl_value_type;

    test_types scalar_values{test_tensorwrapper::smooth_scalar()};
    test_types matrix_values{test_tensorwrapper::smooth_matrix()};

    auto value0 = std::get<object_type>(scalar_values);
    auto value2 = std::get<object_type>(matrix_values);

    dsl::PairwiseParser p;

    SECTION("assignment") {
        object_type rv{};
        object_type corr{};
        SECTION("scalar") {
            p.dispatch(rv(""), value0(""));
            corr.permute_assignment("", value0(""));
            REQUIRE(corr.are_equal(rv));
        }

        SECTION("matrix") {
            p.dispatch(rv("i,j"), value2("i,j"));
            corr.permute_assignment("i,j", value2("i,j"));
            REQUIRE(corr.are_equal(rv));
        }
    }

    SECTION("addition") {
        object_type rv{};
        object_type corr{};
        SECTION("scalar") {
            p.dispatch(rv(""), value0("") + value0(""));
            corr.addition_assignment("", value0(""), value0(""));
            REQUIRE(corr.are_equal(rv));
        }

        SECTION("matrix") {
            p.dispatch(rv("i,j"), value2("i,j") + value2("i,j"));
            corr.addition_assignment("i,j", value2("i,j"), value2("i,j"));
            REQUIRE(corr.are_equal(rv));
        }
    }

    SECTION("subtraction") {
        object_type rv{};
        object_type corr{};
        SECTION("scalar") {
            p.dispatch(rv(""), value0("") - value0(""));
            corr.subtraction_assignment("", value0(""), value0(""));
            REQUIRE(corr.are_equal(rv));
        }

        SECTION("matrix") {
            p.dispatch(rv("i,j"), value2("i,j") - value2("i,j"));
            corr.subtraction_assignment("i,j", value2("i,j"), value2("i,j"));
            REQUIRE(corr.are_equal(rv));
        }
    }

    SECTION("scalar_multiplication") {
        // N.b., only tensor and buffer will override so here we're checking
        // that other objects throw
        using error_t = std::runtime_error;

        REQUIRE_THROWS_AS(p.dispatch(value0(""), value0("") * 1.0), error_t);
    }
}