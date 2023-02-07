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
#include "tiled_array_types.hpp"

namespace tensorwrapper::tensor::allocator::detail_ {

/**
 *  Creates a field-generic TA::TiledRange for a specific tiling scheme
 *
 *  @tparam ShapeType Strong type correponding to shape instance (Field generic)
 *
 *  @param[in] tiling The tiling scheme from which to generate the tiling range
 *  @param[in] shape The shape of the tensor for which to generate the tiling
 *  @returns   TA::TiledRange corresponding to `shape` in the `tiling` scheme
 */
template<typename ShapeType>
tiled_range_type make_tiled_range(const ShapeType& shape) {
    using size_type = typename ShapeType::size_type;

    const auto tiling  = shape.tiling();
    const auto n_modes = tiling.size();
    std::vector<tr1_type> tr1s(n_modes);
    for(size_type mode_i = 0; mode_i < n_modes; ++mode_i) {
        auto tiling_i = tiling[mode_i];
        tr1s[mode_i]  = tr1_type(tiling_i.begin(), tiling_i.end());
    }

    return tiled_range_type(tr1s.begin(), tr1s.end());
}

} // namespace tensorwrapper::tensor::allocator::detail_
