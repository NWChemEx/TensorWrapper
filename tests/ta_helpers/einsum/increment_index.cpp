/*
 * Copyright 2022 NWChemEx-Project
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

#include "tensorwrapper/ta_helpers/einsum/index_utils.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::ta_helpers::einsum;
using tensorwrapper::ta_helpers::einsum::increment_index;

TEST_CASE("increment_index") {
    SECTION("No indices") {
        types::assoc_index idx;
        types::assoc_range ranges;
        bool is_done = increment_index(idx, ranges);
        SECTION("Is done") { REQUIRE(is_done); }
        SECTION("Index value") { REQUIRE(idx == types::assoc_index{}); }
    }

    SECTION("One index") {
        types::assoc_index idx{{"i", 0}};

        SECTION("One element") {
            types::assoc_range ranges{{"i", {0, 1}}};
            types::assoc_index corr{{"i", 0}};
            bool is_done = increment_index(idx, ranges);
            SECTION("Is done") { REQUIRE(is_done); }
            SECTION("Index value") { REQUIRE(idx == corr); }
        }

        SECTION("Two elements") {
            types::assoc_range ranges{{"i", {0, 2}}};
            types::assoc_index corr{{"i", 1}};
            bool is_done = increment_index(idx, ranges);
            SECTION("1st iteration: Is done") { REQUIRE_FALSE(is_done); }
            SECTION("1st iteration: Index value") { REQUIRE(idx == corr); }

            corr    = types::assoc_index{{"i", 0}};
            is_done = increment_index(idx, ranges);
            SECTION("2nd iteration: Is done") { REQUIRE(is_done); }
            SECTION("2nd iteration: Index value") { REQUIRE(idx == corr); }
        }

        SECTION("Three elements") {
            types::assoc_range ranges{{"i", {0, 3}}};
            types::assoc_index corr{{"i", 1}};
            bool is_done = increment_index(idx, ranges);
            SECTION("1st iteration: Is done") { REQUIRE_FALSE(is_done); }
            SECTION("1st iteration: Index value") { REQUIRE(idx == corr); }

            corr    = types::assoc_index{{"i", 2}};
            is_done = increment_index(idx, ranges);
            SECTION("2nd iteration: Is done") { REQUIRE_FALSE(is_done); }
            SECTION("2nd iteration: Index value") { REQUIRE(idx == corr); }

            corr    = types::assoc_index{{"i", 0}};
            is_done = increment_index(idx, ranges);
            SECTION("3rd iteration: Is done") { REQUIRE(is_done); }
            SECTION("3rd iteration: Index value") { REQUIRE(idx == corr); }
        }
    }

    SECTION("Two indices") {
        types::assoc_index idx{{"i", 0}, {"j", 1}};

        SECTION("One element") {
            types::assoc_range ranges{{"i", {0, 1}}, {"j", {1, 2}}};
            types::assoc_index corr{{"i", 0}, {"j", 1}};
            bool is_done = increment_index(idx, ranges);
            SECTION("Is done") { REQUIRE(is_done); }
            SECTION("Index value") { REQUIRE(idx == corr); }
        }

        SECTION("Two elements") {
            SECTION("First index") {
                types::assoc_range ranges{{"i", {0, 2}}, {"j", {1, 2}}};
                types::assoc_index corr{{"i", 1}, {"j", 1}};
                bool is_done = increment_index(idx, ranges);
                SECTION("1st iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("1st iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 0}, {"j", 1}};
                is_done = increment_index(idx, ranges);
                SECTION("2nd iteration: Is done") { REQUIRE(is_done); }
                SECTION("2nd iteration: Index value") { REQUIRE(idx == corr); }
            }

            SECTION("Second index") {
                types::assoc_range ranges{{"i", {0, 1}}, {"j", {1, 3}}};
                types::assoc_index corr{{"i", 0}, {"j", 2}};
                bool is_done = increment_index(idx, ranges);
                SECTION("1st iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("1st iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 0}, {"j", 1}};
                is_done = increment_index(idx, ranges);
                SECTION("2nd iteration: Is done") { REQUIRE(is_done); }
                SECTION("2nd iteration: Index value") { REQUIRE(idx == corr); }
            }
        }

        SECTION("Three elements") {
            SECTION("Two first, one second") {
                types::assoc_range ranges{{"i", {0, 3}}, {"j", {1, 2}}};
                types::assoc_index corr{{"i", 1}, {"j", 1}};
                bool is_done = increment_index(idx, ranges);
                SECTION("1st iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("1st iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 2}, {"j", 1}};
                is_done = increment_index(idx, ranges);
                SECTION("2nd iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("2nd iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 0}, {"j", 1}};
                is_done = increment_index(idx, ranges);
                SECTION("3rd iteration: Is done") { REQUIRE(is_done); }
                SECTION("3rd iteration: Index value") { REQUIRE(idx == corr); }
            }

            SECTION("One first, two second") {
                types::assoc_range ranges{{"i", {0, 1}}, {"j", {1, 4}}};
                types::assoc_index corr{{"i", 0}, {"j", 2}};
                bool is_done = increment_index(idx, ranges);
                SECTION("1st iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("1st iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 0}, {"j", 3}};
                is_done = increment_index(idx, ranges);
                SECTION("2nd iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("2nd iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 0}, {"j", 1}};
                is_done = increment_index(idx, ranges);
                SECTION("3rd iteration: Is done") { REQUIRE(is_done); }
                SECTION("3rd iteration: Index value") { REQUIRE(idx == corr); }
            }
        }

        SECTION("Four elements") {
            SECTION("Two each") {
                types::assoc_range ranges{{"i", {0, 2}}, {"j", {1, 3}}};
                types::assoc_index corr{{"i", 1}, {"j", 1}};
                bool is_done = increment_index(idx, ranges);
                SECTION("1st iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("1st iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 0}, {"j", 2}};
                is_done = increment_index(idx, ranges);
                SECTION("2nd iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("2nd iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 1}, {"j", 2}};
                is_done = increment_index(idx, ranges);
                SECTION("3rd iteration: Is done") { REQUIRE_FALSE(is_done); }
                SECTION("3rd iteration: Index value") { REQUIRE(idx == corr); }

                corr    = types::assoc_index{{"i", 0}, {"j", 1}};
                is_done = increment_index(idx, ranges);
                SECTION("4th iteration: Is done") { REQUIRE(is_done); }
                SECTION("4th iteration: Index value") { REQUIRE(idx == corr); }
            }
        }
    }
}