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
#include <iomanip>
#include <tensorwrapper/utilities/to_json.hpp>

using namespace tensorwrapper;
using namespace testing;

using tensorwrapper::utilities::to_json;

TEMPLATE_LIST_TEST_CASE("to_json", "", std::tuple<double>) {
    Tensor scalar(smooth_scalar_<TestType>());
    Tensor vector(smooth_vector_<TestType>());
    Tensor matrix(smooth_matrix_<TestType>());
    Tensor tensor(smooth_tensor3_<TestType>());

    std::stringstream ss;

    SECTION("scalar") {
        auto pss = &(to_json(ss, scalar));
        REQUIRE(pss == &ss);
        REQUIRE(ss.str() == "42");
    }

    SECTION("vector") {
        auto pss = &(to_json(ss, vector));
        REQUIRE(pss == &ss);
        REQUIRE(ss.str() == "[0,1,2,3,4]");
    }

    SECTION("matrix") {
        auto pss = &(to_json(ss, matrix));
        REQUIRE(pss == &ss);
        REQUIRE(ss.str() == "[[1,2],[3,4]]");
    }

    SECTION("tensor") {
        auto pss = &(to_json(ss, tensor));
        REQUIRE(pss == &ss);
        REQUIRE(ss.str() == "[[[1,2],[3,4]],[[5,6],[7,8]]]");
    }
}
