/*
 * Copyright 2026 NWChemEx-Project
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

#include <tensorwrapper/generate/identity_matrix.hpp>
#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::generate;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;

TEST_CASE("identity_matrix") {
    SECTION("2 by 2") {
        auto result = identity_matrix(2);
        auto corr   = make_tensor({2, 2}, std::vector<double>{1, 0, 0, 1});
        REQUIRE(approximately_equal(result, corr));
    }

    SECTION("3 by 3") {
        auto result = identity_matrix(3);
        auto corr =
          make_tensor({3, 3}, std::vector<double>{1, 0, 0, 0, 1, 0, 0, 0, 1});
        REQUIRE(approximately_equal(result, corr));
    }

    SECTION("invalid n throws") {
        REQUIRE_THROWS_AS(identity_matrix(0), std::invalid_argument);
    }
}
