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

#include "performance_testing.hpp"
#include <iostream>
#include <parallelzone/parallelzone.hpp>

using namespace tensorwrapper;

TEST_CASE("Contraction") {
    Tensor A{1.0, 2.0, 3.0};
    Tensor B{4.0, 5.0, 6.0};
    Tensor C;

    auto l = [&A, &B, &C]() {
        C("") = A("i") * B("i");
        return C;
    };

    parallelzone::hardware::CPU cpu;
    auto [rv, info] = cpu.profile_it(std::move(l));

    std::cout << "Time in ns: " << info.wall_time.count() << std::endl;
}
