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

TEMPLATE_LIST_TEST_CASE("PairwiseParser", "", testing::dsl_types) {
    using object_type = TestType;

    auto scalar_values = testing::scalar_values();
    auto vector_values = testing::vector_values();
    auto matrix_values = testing::matrix_values();

    auto value0 = std::get<object_type>(scalar_values);
    auto value1 = std::get<object_type>(vector_values);
    auto value2 = std::get<object_type>(matrix_values);

    dsl::PairwiseParser p;

    SECTION("assignment") {
        object_type rv(value1);
        object_type corr(value1);
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
        object_type rv(value1);
        object_type corr(value1);
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
        object_type rv(value1);
        object_type corr(value1);
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

    SECTION("multiplication") {
        object_type rv(value1);
        object_type corr(value1);
        SECTION("scalar") {
            p.dispatch(rv(""), value0("") * value0(""));
            corr.multiplication_assignment("", value0(""), value0(""));
            REQUIRE(corr.are_equal(rv));
        }

        SECTION("matrix") {
            p.dispatch(rv("i,j"), value2("i,j") * value2("i,j"));
            corr.multiplication_assignment("i,j", value2("i,j"), value2("i,j"));
            REQUIRE(corr.are_equal(rv));
        }
    }

    SECTION("scalar_multiplication") {
        if constexpr(std::is_same_v<TestType, Tensor>) {
            object_type rv(value1);
            object_type corr(value1);

            SECTION("scalar") {
                p.dispatch(rv(""), value0("") * 2.0);
                corr.scalar_multiplication("", 2.0, value0(""));
                REQUIRE(corr.are_equal(rv));
            }
            SECTION("matrix") {
                p.dispatch(rv("i,j"), value2("i,j") * 2.0);
                corr.scalar_multiplication("i,j", 2.0, value2("i,j"));
                REQUIRE(corr.are_equal(rv));
            }

        } else {
            // N.b., only tensor and buffer will override so here we're checking
            // that other objects throw
            using error_t = std::runtime_error;

            REQUIRE_THROWS_AS(p.dispatch(value0(""), value0("") * 1.0),
                              error_t);
        }
    }
}

// Since Eigen buffers are templated on the rank there isn't an easy way to
// include them in dsl_types
TEST_CASE("PairwiseParser : buffer::Eigen") {
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

    dsl::PairwiseParser p;

    SECTION("assignment") {
        SECTION("scalar") {
            p.dispatch(scalar0(""), scalar1(""));
            corr.permute_assignment("", scalar1(""));
            REQUIRE(corr.are_equal(scalar0));
        }
    }

    SECTION("addition") {
        SECTION("scalar") {
            p.dispatch(scalar0(""), scalar1("") + scalar2(""));
            corr.addition_assignment("", scalar1(""), scalar2(""));
            REQUIRE(corr.are_equal(scalar0));
        }
    }

    SECTION("subtraction") {
        SECTION("scalar") {
            p.dispatch(scalar0(""), scalar1("") - scalar2(""));
            corr.subtraction_assignment("", scalar1(""), scalar2(""));
            REQUIRE(corr.are_equal(scalar0));
        }
    }

    SECTION("multiplication") {
        SECTION("scalar") {
            p.dispatch(scalar0(""), scalar1("") * scalar2(""));
            corr.multiplication_assignment("", scalar1(""), scalar2(""));
            REQUIRE(corr.are_equal(scalar0));
        }
    }

    SECTION("scalar_multiplication") {
        p.dispatch(scalar0(""), scalar1("") * 1.0);
        corr.scalar_multiplication("", 1.0, scalar1(""));
        REQUIRE(corr.are_equal(scalar0));
    }
}
