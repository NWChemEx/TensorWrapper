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

using test_types = std::tuple<Tensor>;

TEMPLATE_LIST_TEST_CASE("Labeled", "", test_types) {
    using object_type  = TestType;
    using labeled_type = dsl::Labeled<TestType>;
    using labels_type  = typename labeled_type::label_type;

    labels_type ij("i,j");
    object_type defaulted{};
    labeled_type labeled_default(defaulted, ij);

    SECTION("Ctor") {
        SECTION("Value") {
            REQUIRE(labeled_default.lhs() == defaulted);
            REQUIRE(labeled_default.rhs() == ij);
        }

        SECTION("to const") {
            using const_labeled_type = dsl::Labeled<const TestType>;
            const_labeled_type const_labeled_default(labeled_default);

            REQUIRE(const_labeled_default.lhs() == defaulted);
            REQUIRE(const_labeled_default.rhs() == ij);
        }
    }

    SECTION("operator=") {
        // At present this operator just calls Parser dispatch. We know that
        // works from other tests so here we just spot check.
        Tensor t;

        // SECTION("scalar") {
        //     Tensor scalar(testing::smooth_scalar());
        //     auto labeled_t  = t("");
        //     auto plabeled_t = &(labeled_t = scalar("") + scalar(""));
        //     REQUIRE(plabeled_t == &labeled_t);

        //     auto buffer      = testing::eigen_scalar<double>();
        //     buffer.value()() = 84.0;
        //     Tensor corr(scalar.logical_layout(), std::move(buffer));
        //     REQUIRE(t == corr);
        // }

        // SECTION("Vector") {
        //     Tensor vector(testing::smooth_vector());
        //     auto labeled_t  = t("i");
        //     auto plabeled_t = &(labeled_t = vector("i") + vector("i"));
        //     REQUIRE(plabeled_t == &labeled_t);

        //     auto buffer = testing::eigen_vector<double>();
        //     for(std::size_t i = 0; i < 5; ++i) buffer.value()(i) = i + i;
        //     Tensor corr(t.logical_layout(), std::move(buffer));
        //     REQUIRE(t == corr);
        // }
    }
}