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

#include "tensorwrapper/ta_helpers/einsum/index_map.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::ta_helpers::einsum;
using tensorwrapper::ta_helpers::einsum::IndexMap;

/* We know splitting strings works from parse_index test so we only need to make
 * sure that the inputs are correctly fed through parse_index and saved
 * correctly.
 */
TEST_CASE("IndexMap::IndexMap(string, string, string)") {
    IndexMap im("i, j, k", "i, k, l", "j, l");

    SECTION("Result indices") {
        types::index_set corr{"i", "j", "k"};
        REQUIRE(im.result_vars() == corr);
    }

    SECTION("LHS indices") {
        types::index_set corr{"i", "k", "l"};
        REQUIRE(im.lhs_vars() == corr);
    }

    SECTION("RHS indices") {
        types::index_set corr{"j", "l"};
        REQUIRE(im.rhs_vars() == corr);
    }
}

TEST_CASE("IndexMap::select_result") {
    IndexMap im("i,j", "j", "i");

    std::map<std::string, std::size_t> qs{{"i", 0}, {"j", 1}, {"k", 2}};

    std::vector<std::size_t> corr{0, 1};
    REQUIRE(im.select_result(qs) == corr);
}

TEST_CASE("IndexMap::select_lhs") {
    IndexMap im("j", "i,j", "i");

    std::map<std::string, std::size_t> qs{{"i", 0}, {"j", 1}, {"k", 2}};

    std::vector<std::size_t> corr{0, 1};
    REQUIRE(im.select_lhs(qs) == corr);
}

TEST_CASE("IndexMap::select_rhs") {
    IndexMap im("i", "j", "j,i");

    std::map<std::string, std::size_t> qs{{"i", 0}, {"j", 1}, {"k", 2}};

    std::vector<std::size_t> corr{1, 0};
    REQUIRE(im.select_rhs(qs) == corr);
}
