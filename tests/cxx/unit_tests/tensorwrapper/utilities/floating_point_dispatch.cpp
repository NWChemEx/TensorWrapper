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

#include "../testing/testing.hpp"

using namespace tensorwrapper;
using namespace tensorwrapper::utilities;

struct Kernel {
    template<typename FloatType>
    void run(buffer::BufferBase& buffer) {
        auto corr = testing::eigen_matrix<FloatType>();
        REQUIRE(corr->are_equal(buffer));
    }

    template<typename FloatType>
    bool run(buffer::BufferBase& buffer, buffer::BufferBase& corr) {
        return corr.are_equal(buffer);
    }
};

TEMPLATE_LIST_TEST_CASE("floating_point_dispatch", "",
                        types::floating_point_types) {
    Kernel kernel;
    auto tensor = testing::eigen_matrix<TestType>();

    SECTION("Single input, no return") {
        floating_point_dispatch(kernel, *tensor);
    }

    SECTION("Two inputs and a return") {
        REQUIRE(floating_point_dispatch(kernel, *tensor, *tensor));
    }
}