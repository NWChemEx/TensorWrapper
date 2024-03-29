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
#include "../../../ta_helpers/get_block_idx.hpp"
#include "tensorwrapper/sparse_map/index.hpp"
#include "tensorwrapper/sparse_map/sparse_map/sparse_map.hpp"
#include "tiling_map_index.hpp"
#include <TiledArray/conversions/make_array.h>
#include <algorithm>
#include <iterator>

namespace tensorwrapper::sparse_map {
namespace detail_ {

/** @brief Removes injected mode offsets from an index.
 *
 *  When dealing with tensors with one or more modes spanned by an independent
 *  index we need to take slices with the independent mode offsets pinned to
 *  the index in the outer tensor. This function removes those pinned modes.
 *
 *  @param[in] idx The index which contains injected mode offsets.
 *  @param[in] injections The map from mode number to injected index, *i.e.*,
 *                        `injections[i]` is the offset to inject as mode `i`.
 *
 * @return A copy of @p idx with injected modes removed.
 *
 * @throw std::runtime_eror if the modes in @p injections are not consistent
 *                          with @p idx. Strong throw guarantee.
 * @throw std::bad_alloc if there is insufficient memory to create the
 *                       uninjected index. Strong throw guarantee.
 */
inline auto uninject_index_(
  const Index& idx, const std::map<std::size_t, std::size_t>& injections) {
    const auto r = idx.size();
    if(injections.empty()) return idx;
    for(const auto& [k, v] : injections)
        if(k >= r)
            throw std::runtime_error("Mode: " + std::to_string(k) +
                                     " is not in range [0, " +
                                     std::to_string(r) + ")");
    std::vector<std::size_t> uninjected_idx(r - injections.size());
    for(std::size_t i = 0, counter = 0; i < r; ++i) {
        if(!injections.count(i)) {
            uninjected_idx[counter] = idx[i];
            ++counter;
        }
    }
    return Index(std::move(uninjected_idx));
}

/** @brief Fills in the provided ToT tile.
 *
 * @tparam TileType The type of a ToT tile. Assumed to satisfy TA's tile
 *                  concept.
 * @tparam T The type of the tensor we are taking elements from. Assumed to be
 *           some variation of TA::SpArray
 *
 * @param[in] tile The initialized tile we are filling in.
 * @param[in] sm The SparseMap guiding how to fill @p tile from @p tensor.
 * @param[in] tensor Where the elements for @p tile will be taken from.
 * @param[in] ind2mode Map from modes of an independent index to mode of
 *                     @p tensor such that `ind2mode[i]` is the mode of
 *                     @p tensor that the `i`-th independent mode maps to.
 * @return The ToT tile after filling it in.
 */
template<typename TileType, typename T>
auto make_tot_tile_(TileType tile, const SparseMap& sm, const T& tensor,
                    const std::map<std::size_t, std::size_t>& ind2mode = {}) {
    using inner_tile_t = typename std::decay_t<TileType>::value_type;

    const auto& trange  = tensor.trange();            // Trange of "dense"
    const auto ind_rank = sm.ind_rank();              // Rank of inner tensor
    const auto inj_rank = ind_rank + ind2mode.size(); // Rank of injected index
    const bool do_inj   = !ind2mode.empty();          // Are we injecting?

    // Move allocations out of loops
    std::map<std::size_t, std::size_t> injections; // Map for injections

    // Loop over outer-elemental indices
    for(const auto& oeidx_v : tile.range()) {
        const Index oeidx(oeidx_v.begin(), oeidx_v.end());

        // Handle scenario where independent index has no domain
        if(!sm.count(oeidx)) {
            tile[oeidx] = inner_tile_t{};
            continue;
        }

        const auto& d = sm.at(oeidx); // The elements in the inner tile
        inner_tile_t buffer(TA::Range(d.result_extents()), 0.0);

        // Determine tiles to retrieve using injected domain
        // TODO: This is just a copy of d if do_inj == false
        for(const auto& [k, v] : ind2mode) injections[v] = oeidx[k];
        auto injected_d = d.inject(injections);
        auto tdomain    = detail_::tile_domain(injected_d, trange);

        for(const auto& itidx : tdomain) { // Loop over inner-tile indices
            if(tensor.is_zero(itidx)) continue;
            inner_tile_t t = tensor.find(itidx);

            // It's not clear to me whether the injection alters the order. If
            // the indices in injected_d are ordered such that the i-th index
            // of injected_d is the i-th index of d (with the former containing
            // the injected modes) then we can loop over injected_d, d zipped
            // together and avoid the uninjection
            for(const auto& ieidx : injected_d) { // Loop over inner-element
                if(t.range().includes(ieidx)) {   // Is element in tile?
                    // Remove injected modes
                    auto lhs_idx = uninject_index_(ieidx, injections);
                    // map it to output
                    buffer[d.result_index(lhs_idx)] = t[ieidx];
                }
            }
        }
        tile[oeidx] = buffer;
    }
    return tile;
}

} // namespace detail_

/** @brief Sparsifies a tensor according to the provided SparseMap.
 *
 *  The most general SparseMap is an element-to-element map. Given such a map,
 *  and the tilings of the tensors, we can make any other sparse map. This
 *  overload of from_sparse_map is at the moment the workhorse for all of the
 *  overloads.
 *
 * @tparam T The type of the tensor being sparsified, assumed to be a TiledArray
 *           tensor.
 *
 * @param[in] esm The element-to-element sparse map describing how to sparsify
 *                the tensor.
 * @param[in] tensor The tensor we are sparsifying.
 * @param[in] outer_trange The TiledRange for the outer tensor of the
 *            tensor-of-tensor this function is creating.
 * @param[in] ind2mode A map from independent index mode to the mode in
 *                     @p tensor it maps to.  *i.e.*, `ind2mode[i]` is the mode
 *                     of @p tensor that the `i`-th mode of an independent index
 *                     maps to.
 * @return The tensor-of-tensors resulting from applying @p esm to @p tensor.
 */
template<typename T>
auto from_sparse_map(const SparseMap& esm, const T& tensor,
                     const TA::TiledRange outer_trange,
                     const std::map<std::size_t, std::size_t>& ind2mode = {}) {
    using scalar_type = typename T::scalar_type;
    using tot_type =
      TA::DistArray<TA::Tensor<TA::Tensor<scalar_type>>, TA::SparsePolicy>;

    if(esm.dep_rank() + ind2mode.size() != tensor.trange().rank())
        throw std::runtime_error("Ranks don't work out.");

    auto tesm = detail_::tile_independent_indices(esm, outer_trange);
    auto l    = [=](auto& tile, const auto& range) {
        using tile_type = std::decay_t<decltype(tile)>;
        auto otidx      = ta_helpers::get_block_idx(outer_trange, range);
        if(!tesm.count(Index(otidx.begin(), otidx.end()))) return 0.0;
        tile = detail_::make_tot_tile_(tile_type(range), esm, tensor, ind2mode);
        return tile.norm();
    };

    return TA::make_array<tot_type>(tensor.world(), outer_trange, l);
}

} // namespace tensorwrapper::sparse_map