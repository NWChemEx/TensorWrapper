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
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace buffer;

/* Testing strategy:
 *
 * - Contiguous is an abstract class. To test it we must create an instance of
 *   a derived class. We then will upcast to Contiguous and perform checks
 *   through the BufferBase interface.

 *
 */

TEMPLATE_LIST_TEST_CASE("buffer::Contiguous", "", types::floating_point_types) {
    using base_type = Contiguous<TestType>;
    auto pt0        = testing::eigen_scalar<TestType>();
    auto pt1        = testing::eigen_vector<TestType>();
    auto& t0        = *pt0;
    auto& t1        = *pt1;

    auto& base0 = static_cast<base_type&>(t0);
    auto& base1 = static_cast<base_type&>(t1);

    SECTION("size") {
        REQUIRE(base0.size() == 1);
        REQUIRE(base1.size() == 5);
    }

    SECTION("data()") {
        REQUIRE(*base0.data() == TestType(42.0));

        REQUIRE(*(base1.data() + 0) == TestType(0.0));
        REQUIRE(*(base1.data() + 1) == TestType(1.0));
        REQUIRE(*(base1.data() + 2) == TestType(2.0));
        REQUIRE(*(base1.data() + 3) == TestType(3.0));
        REQUIRE(*(base1.data() + 4) == TestType(4.0));
    }

    SECTION("data() const") {
        REQUIRE(*std::as_const(base0).data() == TestType(42.0));

        REQUIRE(*(std::as_const(base1).data() + 0) == TestType(0.0));
        REQUIRE(*(std::as_const(base1).data() + 1) == TestType(1.0));
        REQUIRE(*(std::as_const(base1).data() + 2) == TestType(2.0));
        REQUIRE(*(std::as_const(base1).data() + 3) == TestType(3.0));
        REQUIRE(*(std::as_const(base1).data() + 4) == TestType(4.0));
    }

    SECTION("at()") {
        REQUIRE(base0.at() == TestType(42.0));

        REQUIRE(base1.at(0) == TestType(0.0));
        REQUIRE(base1.at(1) == TestType(1.0));
        REQUIRE(base1.at(2) == TestType(2.0));
        REQUIRE(base1.at(3) == TestType(3.0));
        REQUIRE(base1.at(4) == TestType(4.0));

        REQUIRE_THROWS_AS(base0.at(0), std::runtime_error);
        REQUIRE_THROWS_AS(base1.at(0, 1), std::runtime_error);
    }

    SECTION("at()const") {
        REQUIRE(std::as_const(base0).at() == TestType(42.0));

        REQUIRE(std::as_const(base1).at(0) == TestType(0.0));
        REQUIRE(std::as_const(base1).at(1) == TestType(1.0));
        REQUIRE(std::as_const(base1).at(2) == TestType(2.0));
        REQUIRE(std::as_const(base1).at(3) == TestType(3.0));
        REQUIRE(std::as_const(base1).at(4) == TestType(4.0));

        REQUIRE_THROWS_AS(std::as_const(base0).at(0), std::runtime_error);
        REQUIRE_THROWS_AS(std::as_const(base1).at(0, 1), std::runtime_error);
    }
}