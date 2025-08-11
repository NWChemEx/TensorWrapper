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

#include "../testing/helpers.hpp"
#include <tensorwrapper/detail_/view_traits.hpp>

using namespace tensorwrapper::detail_;

TEST_CASE("is_mutable_to_immutable_cast_v") {
    // N.B. Only the const-ness of the types and whether they differ by
    // const-ness should matter
    STATIC_REQUIRE(is_mutable_to_immutable_cast_v<double, const double>);

    STATIC_REQUIRE_FALSE(is_mutable_to_immutable_cast_v<int, const double>);
    STATIC_REQUIRE_FALSE(is_mutable_to_immutable_cast_v<const double, double>);
    STATIC_REQUIRE_FALSE(is_mutable_to_immutable_cast_v<double, double>);
    STATIC_REQUIRE_FALSE(
      is_mutable_to_immutable_cast_v<const double, const double>);
}

TEST_CASE("enable_if_mutable_to_immutable_cast_t") {
    STATIC_REQUIRE(
      std::is_same_v<
        enable_if_mutable_to_immutable_cast_t<double, const double>, void>);
}
