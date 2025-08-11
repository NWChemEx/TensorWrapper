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
#include <tensorwrapper/tensorwrapper.hpp>

using namespace tensorwrapper;

using test_types = std::tuple<shape::Smooth>;

TEMPLATE_LIST_TEST_CASE("Labeled", "", test_types) {
    using object_type        = TestType;
    using labeled_type       = dsl::Labeled<TestType>;
    using const_labeled_type = dsl::Labeled<const TestType>;
    using labels_type        = typename labeled_type::label_type;

    test_types defaulted_values{shape::Smooth{}};
    test_types values{testing::smooth_matrix()};

    labels_type scalar;
    labels_type ij("i,j");
    auto defaulted = std::get<object_type>(defaulted_values);
    auto value     = std::get<object_type>(values);

    labeled_type labeled_default(defaulted, scalar);
    labeled_type labeled_value(value, ij);
    const_labeled_type clabeled_default(defaulted, scalar);
    const_labeled_type clabeled_value(value, ij);

    SECTION("Ctor") {
        SECTION("Value") {
            // Taking label_type object
            REQUIRE(&labeled_default.object() == &defaulted);
            REQUIRE(labeled_default.labels() == scalar);

            REQUIRE(&clabeled_default.object() == &defaulted);
            REQUIRE(clabeled_default.labels() == scalar);

            REQUIRE(&labeled_value.object() == &value);
            REQUIRE(labeled_value.labels() == ij);

            REQUIRE(&clabeled_value.object() == &value);
            REQUIRE(clabeled_value.labels() == ij);

            // Taking string literal object
            labeled_type labeled2(value, "i,j");
            REQUIRE(&labeled2.object() == &value);
            REQUIRE(labeled2.labels() == ij);
        }

        SECTION("mutable to const conversion") {
            const_labeled_type const_labeled_default(labeled_default);

            REQUIRE(&const_labeled_default.object() == &defaulted);
            REQUIRE(const_labeled_default.labels() == scalar);
        }

        // N.b., there is no default ctor so we can't use the testing helpers
        SECTION("Copy ctor") {
            labeled_type labeled_copy(labeled_default);
            REQUIRE(&labeled_copy.object() == &defaulted);
            REQUIRE(&labeled_copy.labels() != &labeled_default.labels());
            REQUIRE(labeled_copy.labels() == scalar);
        }

        SECTION("Move ctor") {
            labeled_type labeled_move(std::move(labeled_value));
            REQUIRE(&labeled_move.object() == &value);
            REQUIRE(labeled_move.labels() == ij);
        }
    }

    SECTION("evaluation, i.e., operator=") {
        // copy-assignment-like operation
        labeled_type other(defaulted, "i,j");
        auto pother = &(other = labeled_value);
        REQUIRE(pother == &other);
        REQUIRE(other.object().are_equal(value));
        REQUIRE(&other.labels() != &labeled_value.labels());
        REQUIRE(other.labels() == "i,j");
    }

    SECTION("object()") {
        REQUIRE(labeled_default.object().are_equal(defaulted));
        REQUIRE(clabeled_default.object().are_equal(defaulted));
        REQUIRE(labeled_value.object().are_equal(value));
        REQUIRE(clabeled_value.object().are_equal(value));
    }

    SECTION("object() const") {
        REQUIRE(std::as_const(labeled_default).object().are_equal(defaulted));
        REQUIRE(std::as_const(clabeled_default).object().are_equal(defaulted));
        REQUIRE(std::as_const(labeled_value).object().are_equal(value));
        REQUIRE(std::as_const(clabeled_value).object().are_equal(value));
    }

    SECTION("labels()") {
        REQUIRE(labeled_default.labels() == scalar);
        REQUIRE(clabeled_default.labels() == scalar);
        REQUIRE(labeled_value.labels() == ij);
        REQUIRE(clabeled_value.labels() == ij);
    }

    SECTION("labels() const") {
        REQUIRE(std::as_const(labeled_default).labels() == scalar);
        REQUIRE(std::as_const(clabeled_default).labels() == scalar);
        REQUIRE(std::as_const(labeled_value).labels() == ij);
        REQUIRE(std::as_const(clabeled_value).labels() == ij);
    }

    SECTION("operator==") {
        // Same values and const-ness
        REQUIRE(labeled_default == labeled_type(defaulted, scalar));

        // Same values different const-ness
        REQUIRE(labeled_default == clabeled_default);
        REQUIRE(clabeled_default == labeled_default);

        // Different object, same labels
        auto value2 = testing::smooth_matrix(20, 10);
        REQUIRE_FALSE(labeled_value == labeled_type(value2, ij));

        // Same object, different labels
        REQUIRE_FALSE(labeled_value == labeled_type(value, "j,i"));
    }

    SECTION("operator!=") {
        // Just negates operator== so spot checking is fine
        REQUIRE_FALSE(labeled_default != clabeled_default);
        REQUIRE(labeled_value != labeled_type(value, "j,i"));
    }
}
