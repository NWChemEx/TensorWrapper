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

#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/operations/norm.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;

TEMPLATE_LIST_TEST_CASE("power", "", types::floating_point_types) {
    SECTION("scalar") {
        shape::Smooth s{};
        Tensor scalar(s, testing::eigen_scalar<TestType>());
        auto rv = power(scalar, 2);
        REQUIRE(approximately_equal(
          rv, Tensor(s, testing::eigen_scalar<TestType>(TestType(42 * 42)))));
    }

    SECTION("vector") {
        shape::Smooth s{5};
        Tensor vector(s, testing::eigen_vector<TestType>());
        auto rv = power(vector, 0.5);
        TestType sqrt2(std::sqrt(2));
        TestType sqrt3(std::sqrt(3));
        std::vector<TestType> data{TestType(0), TestType(1), sqrt2, sqrt3,
                                   TestType(2)};
        auto corr = make_tensor({5}, data.begin(), data.end());
        REQUIRE(approximately_equal(rv, corr));
    }
}
