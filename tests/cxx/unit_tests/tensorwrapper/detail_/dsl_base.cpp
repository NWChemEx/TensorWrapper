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

/* Testing Strategy.
 *
 * Derived classes are responsible for overriding the virtual methods of the
 * DSLBase class and testing that their overloads work by going through at
 * least one public API member. The tests here assume that the virtual method
 * implementations work and test that the various public APIs to access those
 * virtual methods work. For example, both `addition` and `addition_assignment`
 * are
 * implemented in terms of `addition_assignment_`. The derived class should test
 * that `addition_assignment_` works by going through `addition_assignment`, but
 * doesn't need to test that `addition` works because this test case will test
 * that.
 *
 * - The tests here also test assertions that can be caught without knowing more
 *   about the objects, e.g., permute assignment must result in an object with
 *   the same (or fewer) modes.
 */

using namespace tensorwrapper;

using test_types = std::tuple<shape::Smooth>;

TEMPLATE_LIST_TEST_CASE("DSLBase", "", test_types) {
    using object_type = TestType;
    using label_type  = typename object_type::label_type;

    test_types default_values{shape::Smooth{}};
    test_types values{testing::smooth_matrix()};

    auto default_value = std::get<object_type>(default_values);
    auto value         = std::get<object_type>(values);

    SECTION("operator()()") {
        SECTION("string labels") {
            auto ldefaulted = default_value("");
            REQUIRE(&ldefaulted.object() == &default_value);
            REQUIRE(ldefaulted.labels() == "");

            auto lvalue = value("i,j");
            REQUIRE(&lvalue.object() == &value);
            REQUIRE(lvalue.labels() == "i,j");
        }

        SECTION("DummyIndices") {
            auto ldefaulted = default_value(label_type(""));
            REQUIRE(&ldefaulted.object() == &default_value);
            REQUIRE(ldefaulted.labels() == "");

            auto lvalue = value(label_type("i,j"));
            REQUIRE(&lvalue.object() == &value);
            REQUIRE(lvalue.labels() == "i,j");
        }
    }

    SECTION("operator()() const") {
        SECTION("string labels") {
            auto ldefaulted = std::as_const(default_value)("");
            REQUIRE(&ldefaulted.object() == &default_value);
            REQUIRE(ldefaulted.labels() == "");

            auto lvalue = std::as_const(value)("i,j");
            REQUIRE(&lvalue.object() == &value);
            REQUIRE(lvalue.labels() == "i,j");
        }

        SECTION("DummyIndices") {
            auto ldefaulted = std::as_const(default_value)(label_type(""));
            REQUIRE(&ldefaulted.object() == &default_value);
            REQUIRE(ldefaulted.labels() == "");

            auto lvalue = std::as_const(value)(label_type("i,j"));
            REQUIRE(&lvalue.object() == &value);
            REQUIRE(lvalue.labels() == "i,j");
        }
    }

    SECTION("addition_assignment") {
        // N.b., does error checks before calling addition_assignment_. We
        // assume addition_assignment_ works and focus on the error checks
        using error_t = std::runtime_error;
        auto s        = default_value("");
        auto sij      = default_value("i,j");
        auto mij      = value("i,j");
        auto mik      = value("i,k");

        // LHS's indices must match rank
        REQUIRE_THROWS_AS(value.addition_assignment("i,j", sij, s), error_t);

        // RHS's indices must match rank
        REQUIRE_THROWS_AS(value.addition_assignment("i,j", s, sij), error_t);

        // LHS and RHS must be related by a permutation
        REQUIRE_THROWS_AS(value.addition_assignment("i,j", mij, mik), error_t);

        // Output must have <= number of dummy indices
        REQUIRE_THROWS_AS(value.addition_assignment("i,j", s, s), error_t);
    }

    SECTION("subtraction_assignment") {
        // N.b., does error checks before calling addition_assignment_. We
        // assume addition_assignment_ works and focus on the error checks
        using error_t = std::runtime_error;
        auto s        = default_value("");
        auto sij      = default_value("i,j");
        auto mij      = value("i,j");
        auto mik      = value("i,k");

        // LHS's indices must match rank
        REQUIRE_THROWS_AS(value.subtraction_assignment("i,j", sij, s), error_t);

        // RHS's indices must match rank
        REQUIRE_THROWS_AS(value.subtraction_assignment("i,j", s, sij), error_t);

        // LHS and RHS must be related by a permutation
        REQUIRE_THROWS_AS(value.subtraction_assignment("i,j", mij, mik),
                          error_t);

        // Output must have <= number of dummy indices
        REQUIRE_THROWS_AS(value.subtraction_assignment("i,j", s, s), error_t);
    }

    SECTION("multiplication_assignment") {
        // N.b., does error checks before calling addition_assignment_. We
        // assume addition_assignment_ works and focus on the error checks
        using error_t = std::runtime_error;
        auto s        = default_value("");
        auto sij      = default_value("i,j");
        auto mij      = value("i,j");
        auto mik      = value("i,k");

        // LHS's indices must match rank
        REQUIRE_THROWS_AS(value.multiplication_assignment("i,j", sij, s),
                          error_t);

        // RHS's indices must match rank
        REQUIRE_THROWS_AS(value.multiplication_assignment("i,j", s, sij),
                          error_t);
    }

    SECTION("permute_assignment") {
        // N.b., does error checks before calling permute_assignment_. We assume
        // permute_assignment_ works and focus on the error checks
        using error_t = std::runtime_error;
        auto s        = default_value("");
        auto sij      = default_value("i,j");

        // Input's indices must match rank
        REQUIRE_THROWS_AS(value.permute_assignment("i,j", sij), error_t);

        // Output must have <= number of dummy indices
        REQUIRE_THROWS_AS(value.permute_assignment("i,j", s), error_t);
    }

    SECTION("scalar_multiplication") {
        // N.b., only tensor and buffer will override so here we're checking
        // that other objects throw
        if constexpr(std::is_same_v<TestType, Tensor>) {
        } else {
            using error_t = std::runtime_error;
            auto s        = default_value("");
            REQUIRE_THROWS_AS(value.scalar_multiplication("", 1.0, s), error_t);
        }
    }
}