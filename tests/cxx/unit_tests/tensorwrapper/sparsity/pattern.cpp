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
#include "../helpers.hpp"
#include <tensorwrapper/sparsity/pattern.hpp>

using namespace tensorwrapper::testing;
using namespace tensorwrapper::sparsity;

TEST_CASE("Pattern") {
    Pattern defaulted;

    SECTION("Ctors, assignment") {
        SECTION("Default") {}

        test_copy_move_ctor_and_assignment(defaulted);
    }

    SECTION("operator==") { REQUIRE(defaulted == Pattern{}); }

    SECTION("operator!=") {
        // Just spot check because it is implemented in terms of operator==
        REQUIRE_FALSE(defaulted != Pattern{});
    }

    SECTION("addition_assignment_") {
        Pattern p0;

        auto pp0 = &(p0.addition_assignment("", defaulted("")));
        REQUIRE(pp0 == &p0);
        REQUIRE(p0 == defaulted);

        // Throws if labels aren't consistent
        REQUIRE_THROWS_AS(p0.addition_assignment("", defaulted("i")),
                          std::runtime_error);
    }

    SECTION("permute_assignment_") {
        Pattern p0;

        auto pp0 = &(p0.permute_assignment("", defaulted("")));
        REQUIRE(pp0 == &p0);
        REQUIRE(p0 == defaulted);

        // Throws if labels aren't consistent
        REQUIRE_THROWS_AS(p0.permute_assignment("", defaulted("i")),
                          std::runtime_error);
    }
}
