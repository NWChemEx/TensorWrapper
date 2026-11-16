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

#include "../testing/testing.hpp"
#include <stdexcept>
#include <tensorwrapper/types/floating_point.hpp>

using namespace tensorwrapper;

TEMPLATE_LIST_TEST_CASE("construct_uq_type(nominal, minus, plus)", "",
                        types::floating_point_types) {
    using T = TestType;

    // nominal = 1, 0.25 below and 0.5 above, i.e., [0.75, 1.5]
    const double nominal = 1.0, minus = 0.25, plus = 0.5;
    const auto value = types::construct_uq_type<T>(nominal, minus, plus);

    if constexpr(types::is_triplex_v<T>) {
        REQUIRE(value.lower() == 0.75);
        REQUIRE(value.nominal() == 1.0);
        REQUIRE(value.upper() == 1.5);
    } else if constexpr(types::is_uncertain_v<T>) {
        // Uncertain can't be asymmetric, so it takes the larger deviation
        REQUIRE(value.mean() == 1.0);
        REQUIRE(value.sd() == 0.5);
    } else if constexpr(types::is_uq_type_v<T>) {
        // The interval-like types keep the exact bounds
        REQUIRE(types::uq_lower(value) == 0.75);
        REQUIRE(types::uq_upper(value) == 1.5);
    } else {
        REQUIRE(value == T(1.0));
    }

    SECTION("negative deviations throw") {
        REQUIRE_THROWS_AS(types::construct_uq_type<T>(nominal, -0.1, plus),
                          std::domain_error);
        REQUIRE_THROWS_AS(types::construct_uq_type<T>(nominal, minus, -0.1),
                          std::domain_error);
    }
}

#ifdef ENABLE_SIGMA
TEST_CASE("triplex_type") {
    using types::tpdouble;
    using types::tpfloat;

    STATIC_REQUIRE(types::is_triplex_v<tpfloat>);
    STATIC_REQUIRE(types::is_triplex_v<tpdouble>);
    STATIC_REQUIRE(types::is_uq_type_v<tpdouble>);
    STATIC_REQUIRE_FALSE(types::is_interval_v<tpdouble>);
    STATIC_REQUIRE_FALSE(types::is_triplex_v<types::idouble>);
    STATIC_REQUIRE_FALSE(types::is_triplex_v<double>);

    SECTION("construct_uq_type(center, radius) is symmetric") {
        auto value = types::construct_uq_type<tpdouble>(1.0, 0.5);
        REQUIRE(value.lower() == 0.5);
        REQUIRE(value.nominal() == 1.0);
        REQUIRE(value.upper() == 1.5);
    }

    SECTION("uq_center is the nominal value, not the median") {
        tpdouble value(0.0, 0.25, 2.0);
        REQUIRE(types::uq_center(value) == 0.25);
        REQUIRE(types::uq_lower(value) == 0.0);
        REQUIRE(types::uq_upper(value) == 2.0);
    }

    SECTION("registered with WTF") {
        auto kind = wtf::enums::from_string("tpdouble");
        REQUIRE(wtf::enums::to_string(kind) == "tpdouble");
    }
}
#endif
