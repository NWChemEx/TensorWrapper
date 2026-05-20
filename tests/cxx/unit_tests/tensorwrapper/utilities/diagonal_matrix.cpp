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
#include <tensorwrapper/utilities/diagonal_matrix.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;
using namespace testing;

TEMPLATE_LIST_TEST_CASE("diagonal_matrix", "", types::floating_point_types) {
    auto diagonal_values = make_tensor({3}, std::vector<TestType>{1, 2, 3});
    auto corr =
      make_tensor({3, 3}, std::vector<TestType>{1, 0, 0, 0, 2, 0, 0, 0, 3});
    auto result = diagonal_matrix(diagonal_values);
    REQUIRE(approximately_equal(result, corr));
}
