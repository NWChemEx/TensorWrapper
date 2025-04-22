/*
 * Copyright 2025 NWChemEx-Project
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
#include <tensorwrapper/buffer/detail_/hash_utilities.hpp>

using namespace tensorwrapper;
using namespace testing;

using buffer::detail_::hash_utilities::hash_input;
using hash_type = buffer::detail_::hash_utilities::hash_type;

// Make sure we know if this changes for some reason
TEST_CASE("hash_type") { REQUIRE(std::is_same_v<hash_type, std::size_t>); }

// Checking dispatching
TEMPLATE_LIST_TEST_CASE("hash_input", "", types::floating_point_types) {
    using value_type = TestType;
    hash_type seed{0};
    if constexpr(types::is_uncertain_v<TestType>) {
        value_type value(1.0, 1.0);
        hash_input(seed, value);
        hash_type corr{0};
        boost::hash_combine(corr, value.mean());
        boost::hash_combine(corr, value.sd());
        for(const auto& [dep, deriv] : value.deps()) {
            boost::hash_combine(corr, dep);
            boost::hash_combine(corr, deriv);
        }
        REQUIRE(seed == corr);
    } else {
        value_type value(1.0);
        hash_input(seed, value);
        hash_type corr{0};
        boost::hash_combine(corr, value_type{1.0});
        REQUIRE(seed == corr);
    }
}