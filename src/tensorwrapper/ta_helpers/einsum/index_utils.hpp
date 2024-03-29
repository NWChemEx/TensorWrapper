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
#include "types.hpp"

namespace tensorwrapper::ta_helpers::einsum {

/** @brief Splits a string index on `,` characters.
 *
 *  Admittedly TiledArray has support for this via the VariableList class;
 *  however, it won't allow one to specify the same index in the same tensor
 *  (*e.g.*, `T("i,i")` for a trace). This function removes that restriction.
 *
 *  @param[in] idx The string representation of the indices. Individual indices
 *                 are assumed to be delimited by commas.
 *
 *  @return An `std::vector` of the indices such that the 0-th element of the
 *          vector is the 0-th index in @p idx.
 */
types::index_set parse_index(std::string idx);

inline auto initial_index(const types::assoc_range& ranges) {
    types::assoc_index index;
    for(auto&& [k, v] : ranges) index[k] = v.first;
    return index;
}

/** @brief Increments the provided index subject to the provided ranges
 *
 *  This function will take an associative index (map from string indices to
 *  integral values) and update it using the provided associative ranges (map
 *  from a string index to a pair of integral values such that the first value
 *  of the pair is the beginning of the range and the second value of the pair
 *  is the just past the end of the range). Indices are updated in column-major
 *  order (the first index runs fast).
 *
 *  @param[in,out] idx The index to increment. It will be incremented in place
 *                     (i.e., the value will be mutated). It will be reset to
 *                     the first value if iteration has completed.
 *  @param[in] ranges  The range each index spans. Each range is assumed to be
 *                     half-open such that the beginning value is in the range,
 *                     but the ending value is just outside the range.
 *  @return `true` if iteration has completed and `false` otherwise.
 */
inline bool increment_index(types::assoc_index& idx,
                            const types::assoc_range& ranges) {
    // Loop over indices looking for one we can increment
    for(auto&& [idx_str, idx_val] : idx) {
        auto&& range = ranges.at(idx_str); // Throws if idx_str not in ranges

        if(idx_val + 1 < range.second) { // Can increment this index, so do it
            idx[idx_str] = idx_val + 1;
            return false; // Not done since we could increment this index
        } else {          // Can't increment it, so reset it
            idx[idx_str] = range.first;
        }
    }

    return true; // We apparently just reset every index, so we're done
}

} // namespace tensorwrapper::ta_helpers::einsum