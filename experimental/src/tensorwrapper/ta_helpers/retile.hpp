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
#include "ta_headers.hpp"
#include "tensorwrapper/sparse_map/index.hpp"

namespace tensorwrapper::ta_helpers {

using sparse_map::Index;

/** @brief This function retiles a TiledRange so that the provided elements lie
 *         on tile boundaries (if they do not already).
 *
 *  Given a list of element indices and an input TiledRange, this function will
 *  create a new TiledRange object which, in addition to the input tilings, also
 *  has tile boundaries at the specified elements.
 *
 *  @param[in] tr The TiledRange we are adding tile boundaries to.
 *  @param[in] boundaries The elements which will fall on tile boundaries in the
 *                        new TiledRange.
 *
 *  @return A TiledRange whose boundary elements are the union of the boundaries
 *          in @p tr and the elements in @p boundaries.
 */
inline TA::TiledRange insert_tile_boundaries(
  const TA::TiledRange& tr, const std::vector<Index>& boundaries) {
    const auto rank = tr.rank();

    // Make the sorted union of the desired boundaries and the current
    // boundaries, asserting all input indices are of the correct rank
    std::set<Index> bounds2;
    for(const auto& x : boundaries) {
        TA_ASSERT(x.size() == rank);
        bounds2.insert(x);
    }

    for(const auto& tile : tr.tiles_range()) {
        const auto trange  = tr.tile(tile);
        const auto lobound = trange.lobound();
        const auto upbound = trange.upbound();
        bounds2.insert(Index(lobound.begin(), lobound.end()));
        bounds2.insert(Index(upbound.begin(), upbound.end()));
    }

    // If you think of bounds 2 as a matrix (rows are index number, columns are
    // modes of the element indices) this is the opposite of what we need for
    // making a TiledRange. Here we "transpose" bounds2
    std::vector<std::set<std::size_t>> new_tiling(rank);
    for(const auto& x : bounds2) {
        for(std::size_t i = 0; i < rank; ++i) new_tiling[i].insert(x[i]);
    }

    // Now we turn that into a vector of TiledRange1 instances
    std::vector<TA::TiledRange1> new_tr1s;
    for(const auto& x : new_tiling) {
        // TiledRange1 requires the iterator to be random access
        std::vector<std::size_t> copy_x(x.begin(), x.end());
        new_tr1s.emplace_back(TA::TiledRange1(copy_x.begin(), copy_x.end()));
    }
    return TA::TiledRange(new_tr1s.begin(), new_tr1s.end());
}

/** @brief Convenience function for calling insert_tile_boundaries when the
 *         desired boundaries are not already in an std::vector.
 *
 *   This function is a thin wrapper around the
 *   `insert_tile_boundaries(TiledRange, std::vector)` overload which, instead
 *   of taking an `std::vector` of `ElementIndex` instances, takes an arbitrary
 *   number of `ElementIndex` instances.
 *
 * @tparam Args Template parameter pack. All types in the parameter pack need to
 *              be implicitly convertible to ElementIndex.
 *
 * @param[in] tr The TiledRange we are adding the tile boundaries to.
 * @param[in] e0 The first new tile boundary to add.
 * @param[in] args The optional second, third, etc. tile boundaries to add/
 *
 * @return The TiledRange instance containing the original boundaries as well as
 *         the boundaries the user specified.
 */
template<typename... Args>
TA::TiledRange insert_tile_boundaries(const TA::TiledRange& tr,
                                      const sparse_map::Index& e0,
                                      Args&&... args) {
    using vector_type = std::vector<sparse_map::Index>;
    const vector_type boundaries{e0, std::forward<Args>(args)...};
    return insert_tile_boundaries(tr, boundaries);
}

} // namespace tensorwrapper::ta_helpers
