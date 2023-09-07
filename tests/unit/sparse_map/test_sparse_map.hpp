/*
 * Copyright 2022 NWChemEx-Project
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

#pragma once
#include "tensorwrapper/sparse_map/sparse_map.hpp"
#include <catch2/catch.hpp>

namespace testing {

template<std::size_t N>
auto make_indices() {
    if constexpr(N == 1) {
        tensorwrapper::sparse_map::Index i0{0}, i1{1}, i2{2}, i3{3}, i4{4};
        return std::make_tuple(i0, i1, i2, i3, i4);
    } else if constexpr(N == 2) {
        tensorwrapper::sparse_map::Index i00{0, 0}, i01{0, 1}, i10{1, 0},
          i11{1, 1};
        return std::make_tuple(i00, i01, i10, i11);
    } else {
        static_assert(N == 1, "Only support rank 1 and 2 indices");
    }
}

} // namespace testing
