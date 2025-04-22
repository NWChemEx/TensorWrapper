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
 * - BufferBase is an abstract class. To test it we must create an instance of
 *   a derived class. We then will upcast to BufferBase and perform checks
 *   through the BufferBase interface.
 * - `xxx_assignment` methods are tested in the derived classes; however, the
 *   corresponding `xxx` method is defined in BufferBase and thus is tested
 *   here (`xxx` being `addition`, `subtraction`, etc.).
 *
 */

TEST_CASE("BufferBase") {
    auto pscalar = testing::eigen_scalar<double>();
    auto& scalar = *pscalar;
    scalar.set_elem({}, 1.0);

    auto pvector = testing::eigen_vector<double>(2);
    auto& vector = *pvector;

    vector.set_elem({0}, 1.0);
    vector.set_elem({1}, 2.0);

    auto scalar_layout = testing::scalar_physical();
    auto vector_layout = testing::vector_physical(2);

    buffer::Eigen<double> defaulted;
    BufferBase& defaulted_base = defaulted;
    BufferBase& scalar_base    = scalar;
    BufferBase& vector_base    = vector;

    SECTION("has_layout") {
        REQUIRE_FALSE(defaulted_base.has_layout());
        REQUIRE(scalar_base.has_layout());
        REQUIRE(vector_base.has_layout());
    }

    SECTION("has_allocator") { REQUIRE_FALSE(defaulted_base.has_allocator()); }

    SECTION("layout") {
        REQUIRE_THROWS_AS(defaulted_base.layout(), std::runtime_error);
        REQUIRE(scalar_base.layout().are_equal(scalar_layout));
        REQUIRE(vector_base.layout().are_equal(vector_layout));
    }

    SECTION("allocator()") {
        REQUIRE_THROWS_AS(defaulted_base.allocator(), std::runtime_error);
    }

    SECTION("allocator() const") {
        REQUIRE_THROWS_AS(std::as_const(defaulted_base).allocator(),
                          std::runtime_error);
    }

    SECTION("operator==") {
        // Defaulted layout == defaulted layout
        REQUIRE(defaulted_base == buffer::Eigen<double>{});

        // Defaulted layout != non-defaulted layout
        REQUIRE_FALSE(defaulted_base == scalar_base);

        // Non-defaulted layout different value
        REQUIRE_FALSE(scalar_base == vector_base);
    }

    SECTION("operator!=") {
        // Just spot check because it negates operator==, which was tested
        REQUIRE(defaulted_base != scalar_base);
        REQUIRE_FALSE(defaulted_base != buffer::Eigen<double>());
    }
}
