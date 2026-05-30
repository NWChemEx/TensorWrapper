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

#include <tensorwrapper/generate/generate_utils.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper::generate;

TEST_CASE("generate_utils") {
    SECTION("make_rng is deterministic for fixed seed") {
        auto gen1 = make_rng(42);
        auto gen2 = make_rng(42);
        REQUIRE(gen1() == gen2());
    }

    SECTION("require_valid_n throws for invalid n") {
        REQUIRE_THROWS_AS(require_valid_n(0), std::invalid_argument);
        REQUIRE_THROWS_AS(require_valid_n(11), std::invalid_argument);
        REQUIRE_NOTHROW(require_valid_n(1));
        REQUIRE_NOTHROW(require_valid_n(10));
    }
}
