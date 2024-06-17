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
#include <TiledArray/tensor.h>

namespace testing {

inline TA::TiledRange single_element_tiles(const std::vector<std::size_t>& ex) {
    std::vector<TA::TiledRange1> trs;
    for(auto e : ex) {
        std::vector<std::size_t> r(e + 1);
        std::iota(r.begin(), r.end(), 0);
        trs.emplace_back(r.begin(), r.end());
    }
    return TA::TiledRange(trs.begin(), trs.end());
}
inline TA::TiledRange one_big_tile(const std::vector<std::size_t>& ex) {
    std::vector<TA::TiledRange1> trs;
    for(auto e : ex) {
        std::vector<std::size_t> r = {0, e};
        trs.emplace_back(r.begin(), r.end());
    }
    return TA::TiledRange(trs.begin(), trs.end());
}

} // namespace testing
