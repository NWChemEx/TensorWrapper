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
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;

TEMPLATE_LIST_TEST_CASE("make_tensor", "", types::floating_point_types) {
    SECTION("scalar") {
        std::vector<TestType> data{42};
        auto tensor  = make_tensor({}, data.begin(), data.end());
        auto tensor2 = make_tensor({}, data);
        Tensor corr(shape::Smooth{}, testing::eigen_scalar<TestType>(42));
        REQUIRE(approximately_equal(tensor, corr));
        REQUIRE(approximately_equal(tensor2, corr));
    }

    SECTION("vector") {
        std::vector<TestType> data{0, 1, 2, 3, 4};
        auto tensor  = make_tensor({5}, data.begin(), data.end());
        auto tensor2 = make_tensor({5}, data);
        Tensor corr(shape::Smooth{5}, testing::eigen_vector<TestType>());
        REQUIRE(approximately_equal(tensor, corr));
        REQUIRE(approximately_equal(tensor2, corr));
    }

    SECTION("matrix") {
        std::vector<TestType> data{1, 2, 3, 4};
        auto tensor  = make_tensor({2, 2}, data.begin(), data.end());
        auto tensor2 = make_tensor({2, 2}, data);
        Tensor corr(shape::Smooth{2, 2}, testing::eigen_matrix<TestType>());
        REQUIRE(approximately_equal(tensor, corr));
        REQUIRE(approximately_equal(tensor2, corr));
    }
    SECTION("tensor3") {
        std::vector<TestType> data{1, 2, 3, 4, 5, 6, 7, 8};
        auto tensor  = make_tensor({2, 2, 2}, data.begin(), data.end());
        auto tensor2 = make_tensor({2, 2, 2}, data);
        Tensor corr(shape::Smooth{2, 2, 2}, testing::eigen_tensor3<TestType>());
        REQUIRE(approximately_equal(tensor, corr));
        REQUIRE(approximately_equal(tensor2, corr));
    }
    SECTION("tensor4") {
        std::vector<TestType> data{1, 2,  3,  4,  5,  6,  7,  8,
                                   9, 10, 11, 12, 13, 14, 15, 16};
        auto tensor  = make_tensor({2, 2, 2, 2}, data.begin(), data.end());
        auto tensor2 = make_tensor({2, 2, 2, 2}, data);
        Tensor corr(shape::Smooth{2, 2, 2, 2},
                    testing::eigen_tensor4<TestType>());
        REQUIRE(approximately_equal(tensor, corr));
        REQUIRE(approximately_equal(tensor2, corr));
    }
}
