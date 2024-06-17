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
#include "block_kernel.hpp"
#include "tensor_kernel.hpp"
#include "types.hpp"

namespace tensorwrapper::ta_helpers::einsum {

inline auto add_tiled_ranges(std::map<types::index, TA::TiledRange1> ranges,
                             const types::index_set& vars,
                             const TA::TiledRange& trange) {
    for(std::size_t i = 0; i < vars.size(); ++i)
        ranges[vars[i]] = trange.dim(i);
    return ranges;
}

template<typename TileType, typename PolicyType>
auto einsum(const types::index& result_idx, const types::index& lhs_idx,
            const types::index& rhs_idx,
            const TA::DistArray<TileType, PolicyType>& lhs,
            const TA::DistArray<TileType, PolicyType>& rhs) {
    // Figure out what sort of operation the user wants
    IndexMap im(result_idx, lhs_idx, rhs_idx);

    // Figure out the TiledRange1 instances for each index
    std::map<types::index, TA::TiledRange1> ranges;
    ranges = add_tiled_ranges(std::move(ranges), im.lhs_vars(), lhs.trange());
    ranges = add_tiled_ranges(std::move(ranges), im.rhs_vars(), rhs.trange());

    // This lambda is the block kernel for the tensor kernel (there's probably a
    // way to pass block_kernel directly, but I gave up...)
    auto l = [](const IndexMap& indices, const types::assoc_range& ranges,
                const TileType& lhs, const TileType& rhs) {
        return block_kernel(indices, ranges, lhs, rhs);
    };

    // Call the tensor kernel and return the result
    return tensor_kernel(im, ranges, lhs, rhs, l);
}

} // namespace tensorwrapper::ta_helpers::einsum