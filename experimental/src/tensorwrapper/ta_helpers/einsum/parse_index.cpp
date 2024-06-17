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

#include "index_utils.hpp"
#include <algorithm> // For replace
#include <iterator>  // For istream_iterator
#include <sstream>   // For istringstream

namespace tensorwrapper::ta_helpers::einsum {

types::index_set parse_index(std::string idx) {
    // 1. Remove all existing spaces
    idx.erase(std::remove(idx.begin(), idx.end(), ' '), idx.end());

    // 2. Switch commas to spaces
    std::replace(idx.begin(), idx.end(), ',', ' ');

    // 3. Use stringstream to split on spaces
    std::istringstream iss(idx);

    // 4. Copy stream contents into index_set and return
    using iss_itr = std::istream_iterator<std::string>;
    return types::index_set(iss_itr(iss), iss_itr{});
}

} // namespace tensorwrapper::ta_helpers::einsum