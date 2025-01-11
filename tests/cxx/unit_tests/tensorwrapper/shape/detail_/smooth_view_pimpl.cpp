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

/* Testing Strategy.
 *
 * At present the only thing actually implemented in SmoothViewPIMPL is
 * are_equal so that's all this test case tests.
 */

using namespace tensorwrapper::shape;

using types2test = std::pair<Smooth, const Smooth>;

TEMPLATE_LIST_TEST_CASE("SmoothViewPIMPL", "", types2test) {
    using pimpl_type = detail_::SmoothAlias<TestType>;
    using shape_type = std::decay_t<TestType>;
    shape_type scalar_shape{}, shape{1, 2, 3};

    pimpl_type scalar(scalar_shape);
    pimpl_type value(shape);

    SECTION("are_equal") {
        SECTION("Same") {
            REQUIRE(scalar.are_equal(pimpl_type(scalar_shape)));
            REQUIRE(value.are_equal(pimpl_type(shape)));
        }

        SECTION("Different rank") {
            shape_type rhs_shape{1};
            pimpl_type rhs(rhs_shape);
            REQUIRE_FALSE(scalar.are_equal(rhs));
        }

        SECTION("Different extents") {
            shape_type rhs_shape{2, 1, 3};
            pimpl_type rhs(rhs_shape);
            REQUIRE_FALSE(value.are_equal(rhs));
        }
    }
}