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

    SECTION("get_mutable_data()") {
        REQUIRE(*base0.get_mutable_data() == TestType(42.0));

        REQUIRE(*(base1.get_mutable_data() + 0) == TestType(0.0));
        REQUIRE(*(base1.get_mutable_data() + 1) == TestType(1.0));
        REQUIRE(*(base1.get_mutable_data() + 2) == TestType(2.0));
        REQUIRE(*(base1.get_mutable_data() + 3) == TestType(3.0));
        REQUIRE(*(base1.get_mutable_data() + 4) == TestType(4.0));
    }

    SECTION("get_immutable_data() const") {
        REQUIRE(*std::as_const(base0).get_immutable_data() == TestType(42.0));

        REQUIRE(*(std::as_const(base1).get_immutable_data() + 0) ==
                TestType(0.0));
        REQUIRE(*(std::as_const(base1).get_immutable_data() + 1) ==
                TestType(1.0));
        REQUIRE(*(std::as_const(base1).get_immutable_data() + 2) ==
                TestType(2.0));
        REQUIRE(*(std::as_const(base1).get_immutable_data() + 3) ==
                TestType(3.0));
        REQUIRE(*(std::as_const(base1).get_immutable_data() + 4) ==
                TestType(4.0));
    }

    SECTION("get_elem() const") {
        REQUIRE(base0.get_elem({}) == TestType(42.0));

        REQUIRE(base1.get_elem({0}) == TestType(0.0));
        REQUIRE(base1.get_elem({1}) == TestType(1.0));
        REQUIRE(base1.get_elem({2}) == TestType(2.0));
        REQUIRE(base1.get_elem({3}) == TestType(3.0));
        REQUIRE(base1.get_elem({4}) == TestType(4.0));

        REQUIRE_THROWS_AS(base0.get_elem({0}), std::runtime_error);
    }

    SECTION("set_elem() const") {
        base0.set_elem({}, TestType(43.0));
        REQUIRE(base0.get_elem({}) == TestType(43.0));

        base1.set_elem({0}, TestType(43.0));
        REQUIRE(base1.get_elem({0}) == TestType(43.0));

        REQUIRE_THROWS_AS(base0.set_elem({0}, TestType{0.0}),
                          std::runtime_error);
    }

    SECTION("get_data() const") {
        REQUIRE(base0.get_data(0) == TestType(42.0));

        REQUIRE(base1.get_data(0) == TestType(0.0));
        REQUIRE(base1.get_data(1) == TestType(1.0));
        REQUIRE(base1.get_data(2) == TestType(2.0));
        REQUIRE(base1.get_data(3) == TestType(3.0));
        REQUIRE(base1.get_data(4) == TestType(4.0));

        REQUIRE_THROWS_AS(base0.get_data(1), std::runtime_error);
    }

    SECTION("set_data() const") {
        base0.set_data(0, TestType(43.0));
        REQUIRE(base0.get_elem({}) == TestType(43.0));

        REQUIRE_THROWS_AS(base0.set_data(1, TestType{0.0}), std::runtime_error);
    }

    SECTION("fill()") {
        base1.fill(TestType{43.0});
        REQUIRE(base1.get_data(0) == TestType(43.0));
        REQUIRE(base1.get_data(1) == TestType(43.0));
        REQUIRE(base1.get_data(2) == TestType(43.0));
        REQUIRE(base1.get_data(3) == TestType(43.0));
        REQUIRE(base1.get_data(4) == TestType(43.0));
    }

    SECTION("copy()") {
        auto data = std::vector<TestType>(5, TestType(43.0));
        base1.copy(data);
        REQUIRE(base1.get_data(0) == TestType(43.0));
        REQUIRE(base1.get_data(1) == TestType(43.0));
        REQUIRE(base1.get_data(2) == TestType(43.0));
        REQUIRE(base1.get_data(3) == TestType(43.0));
        REQUIRE(base1.get_data(4) == TestType(43.0));
    }
}
