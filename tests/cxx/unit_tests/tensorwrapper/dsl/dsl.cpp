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

    auto scalar_values  = testing::scalar_values();
    auto vector_values  = testing::vector_values();
    auto matrix_values  = testing::matrix_values();
    auto tensor4_values = testing::tensor4_values();

    auto value0 = std::get<object_type>(scalar_values);
    auto value1 = std::get<object_type>(vector_values);
    auto value2 = std::get<object_type>(matrix_values);
    auto value4 = std::get<object_type>(tensor4_values);

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

        value0("m,n") = value2("l,s") * value4("m,n,s,l");
        value1.multiplication_assignment("m,n", value2("l,s"),
                                         value4("m,n,s,l"));
        REQUIRE(value1.are_equal(value0));
    }

    SECTION("scalar_multiplication") {
        if constexpr(std::is_same_v<TestType, Tensor>) {
        } else {
            // N.b., only tensor and buffer will override so here we're checking
            // that other objects throw
            using error_t = std::runtime_error;

            REQUIRE_THROWS_AS(value0("") = value0("") * 1.0, error_t);
        }
    }
}

// Since Eigen buffers are templated on the rank there isn't an easy way to
// include them in dsl_types
TEST_CASE("DSLr : buffer::Eigen") {
    auto pscalar0 = testing::eigen_scalar<float>();
    auto pscalar1 = testing::eigen_scalar<float>();
    auto pscalar2 = testing::eigen_scalar<float>();
    auto pcorr    = testing::eigen_scalar<float>();
    auto& scalar0 = *pscalar0;
    auto& scalar1 = *pscalar1;
    auto& scalar2 = *pscalar2;
    auto& corr    = *pcorr;

    scalar0.set_data(0, 1.0);
    scalar1.set_data(0, 2.0);
    scalar2.set_data(0, 3.0);

    SECTION("assignment") {
        SECTION("scalar") {
            scalar0("") = scalar1("");
            corr.permute_assignment("", scalar1(""));
            REQUIRE(corr.are_equal(scalar0));
        }
    }

    SECTION("addition") {
        SECTION("scalar") {
            scalar0("") = scalar1("") + scalar2("");
            corr.addition_assignment("", scalar1(""), scalar2(""));
            REQUIRE(corr.are_equal(scalar0));
        }
    }

    SECTION("subtraction") {
        SECTION("scalar") {
            scalar0("") = scalar1("") - scalar2("");
            corr.subtraction_assignment("", scalar1(""), scalar2(""));
            REQUIRE(corr.are_equal(scalar0));
        }
    }

    SECTION("multiplication") {
        SECTION("scalar") {
            scalar0("") = scalar1("") * scalar2("");
            corr.multiplication_assignment("", scalar1(""), scalar2(""));
            REQUIRE(corr.are_equal(scalar0));
        }
    }

    SECTION("scalar_multiplication") {
        scalar0("") = scalar1("") * 1.0;
        corr.scalar_multiplication("", 1.0, scalar1(""));
        REQUIRE(corr.are_equal(scalar0));
    }
}