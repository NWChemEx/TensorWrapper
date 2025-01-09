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

#include "../../testing/testing.hpp"
#include <tensorwrapper/shape/detail_/smooth_alias.hpp>

using namespace tensorwrapper::shape;

using types2test = std::pair<Smooth, const Smooth>;

TEMPLATE_LIST_TEST_CASE("SmoothAlias", "", types2test) {
    using pimpl_type = detail_::SmoothAlias<TestType>;
    std::decay_t<TestType> scalar_shape{}, shape{1, 2, 3};

    pimpl_type scalar(scalar_shape);
    pimpl_type value(shape);

    SECTION("CTor") {
        REQUIRE(scalar.rank() == scalar_shape.rank());
        REQUIRE(scalar.size() == scalar_shape.size());

        REQUIRE(value.rank() == shape.rank());
        REQUIRE(value.size() == shape.size());
    }

    SECTION("clone") {
        REQUIRE(scalar.clone()->are_equal(scalar));
        REQUIRE(value.clone()->are_equal(value));
    }

    SECTION("extent") {
        REQUIRE_THROWS_AS(scalar.extent(0), std::out_of_range);
        REQUIRE(value.extent(0) == 1);
        REQUIRE(value.extent(1) == 2);
        REQUIRE(value.extent(2) == 3);
    }

    SECTION("rank") {
        REQUIRE(scalar.rank() == 0);
        REQUIRE(value.rank() == 3);
    }

    SECTION("size") {
        REQUIRE(scalar.size() == 1);
        REQUIRE(value.size() == 6);
    }
}