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
#include "tensorwrapper/sparse_map/index.hpp"
#include "tensorwrapper/sparse_map/sparse_map/sparse_map.hpp"
#include <TiledArray/tiled_range.h>

namespace tensorwrapper::sparse_map::detail_ {

/** @brief Returns a sparse map with independent elements based on tiles.
 *
 *  Produce a new sparse map whose independent elements are the independent
 *  elements of another map converted to tile indices based on the provided
 *  tiled range.
 *
 *  @param[in] sm The original sparse map
 *  @param[in] tr The tiled range used to convert the elements
 *
 *  @return The new sparse map with tile based independent indices.
 *
 *  @throw std::runtime_error if the rank of the tiled range does not match the
 *                            rank of the independent indices.
 */
inline sparse_map::SparseMap tile_independent_indices(
  const sparse_map::SparseMap& sm, const TA::TiledRange& tr) {
    if(tr.rank() != sm.ind_rank())
        throw std::runtime_error("Rank of TiledRange does not equal"
                                 "independent index rank");
    sparse_map::SparseMap new_sm;
    for(const auto& [ind_idx, d] : sm) {
        auto temp = tr.tiles_range().idx(tr.element_to_tile(ind_idx));
        sparse_map::Index new_idx(temp.begin(), temp.end());
        for(const auto& dep_idx : d) { new_sm.add_to_domain(new_idx, dep_idx); }
    }
    return new_sm;
}

/** @brief Returns a sparse map with dependent elements based on tiles.
 *
 *  Produce a new sparse map whose dependent elements are the dependent
 *  elements of another map converted to tile indices based on the provided
 *  tiled range.
 *
 *  @param[in] sm The original sparse map
 *  @param[in] tr The tiled range used to convert the elements
 *
 *  @return The new sparse map with tile based dependent indices.
 *
 *  @throw std::runtime_error if the rank of the tiled range does not match the
 *                            rank of the dependent indices.
 */
inline sparse_map::SparseMap tile_dependent_indices(
  const sparse_map::SparseMap& sm, const TA::TiledRange& tr) {
    if(tr.rank() != sm.dep_rank())
        throw std::runtime_error("Rank of TiledRange does not equal"
                                 "dependent index rank");
    sparse_map::SparseMap new_sm;
    for(const auto& [ind_idx, d] : sm) {
        for(const auto& dep_idx : d) {
            auto temp = tr.tiles_range().idx(tr.element_to_tile(dep_idx));
            sparse_map::Index new_idx(temp.begin(), temp.end());
            new_sm.add_to_domain(ind_idx, new_idx);
        }
    }
    return new_sm;
}

/** @brief Returns a sparse map with elements based on tiles.
 *
 *  Produce a new sparse map whose elements are the elements of another map
 *  converted to tile indices based on the provided tiled range.
 *
 *  @param[in] sm The original sparse map
 *  @param[in] ind_tr The tiled range used to convert the independent elements
 *  @param[in] dep_tr The tiled range used to convert the dependent elements
 *
 *  @return The new sparse map with tile based indices.
 *
 *  @throw std::runtime_error if the ranks of the tiled range does not match the
 *                            ranks of the indices.
 */
inline sparse_map::SparseMap tile_indices(const sparse_map::SparseMap& sm,
                                          const TA::TiledRange& ind_tr,
                                          const TA::TiledRange& dep_tr) {
    auto intermediate_sm = tile_independent_indices(sm, ind_tr);
    return tile_dependent_indices(intermediate_sm, dep_tr);
}

/** @brief Returns a sparse map with independent elements converted from tile to
 *         element indices
 *
 *  Produce a new sparse map whose independent indices are the independent
 *  indices of another map converted to elements based on the provided
 *  tiled range.
 *
 *  @param[in] sm The original sparse map
 *  @param[in] tr The tiled range used to convert the elements
 *
 *  @return The new sparse map with element based independent indices.
 *
 *  @throw std::runtime_error if the rank of the tiled range does not match the
 *                            rank of the independent indices.
 */
inline sparse_map::SparseMap untile_independent_indices(
  const sparse_map::SparseMap& sm, const TA::TiledRange& tr) {
    if(tr.rank() != sm.ind_rank())
        throw std::runtime_error("Rank of TiledRange does not equal"
                                 "independent index rank");
    sparse_map::SparseMap new_sm;
    for(const auto& [ind_idx, d] : sm) {
        for(const auto& temp : tr.make_tile_range(ind_idx)) {
            for(const auto& dep_idx : d) {
                sparse_map::Index new_idx(temp.begin(), temp.end());
                new_sm.add_to_domain(new_idx, dep_idx);
            }
        }
    }
    return new_sm;
}

/** @brief Returns a sparse map with dependent elements converted from tile to
 *         element indices
 *
 *  Produce a new sparse map whose dependent indices are the dependent
 *  indices of another map converted to elements based on the provided
 *  tiled range.
 *
 *  @param[in] sm The original sparse map
 *  @param[in] tr The tiled range used to convert the elements
 *
 *  @return The new sparse map with element based dependent indices.
 *
 *  @throw std::runtime_error if the rank of the tiled range does not match the
 *                            rank of the dependent indices.
 */
inline sparse_map::SparseMap untile_dependent_indices(
  const sparse_map::SparseMap& sm, const TA::TiledRange& tr) {
    if(tr.rank() != sm.dep_rank())
        throw std::runtime_error("Rank of TiledRange does not equal"
                                 "dependent index rank");
    sparse_map::SparseMap new_sm;
    for(const auto& [ind_idx, d] : sm) {
        for(const auto& dep_idx : d) {
            for(const auto& temp : tr.make_tile_range(dep_idx)) {
                sparse_map::Index new_idx(temp.begin(), temp.end());
                new_sm.add_to_domain(ind_idx, new_idx);
            }
        }
    }
    return new_sm;
}

/** @brief Returns a sparse map with indices based on tensor elements.
 *
 *  Produce a new sparse map whose indices are the indices of another map
 *  converted to elements based on the provided tiled range.
 *
 *  @param[in] sm The original sparse map
 *  @param[in] ind_tr The tiled range used to convert the independent elements
 *  @param[in] dep_tr The tiled range used to convert the dependent elements
 *
 *  @return The new sparse map with element based indices.
 *
 *  @throw std::runtime_error if the ranks of the tiled range does not match the
 *                            ranks of the indices.
 */
inline sparse_map::SparseMap untile_indices(const sparse_map::SparseMap& sm,
                                            const TA::TiledRange& ind_tr,
                                            const TA::TiledRange& dep_tr) {
    auto intermediate_sm = untile_independent_indices(sm, ind_tr);
    return untile_dependent_indices(intermediate_sm, dep_tr);
}

inline sparse_map::Domain tile_domain(const Domain& d,
                                      const TA::TiledRange& trange) {
    sparse_map::Domain new_domain;
    const auto& erange = trange.elements_range();
    for(const auto& x : d) {
        if(!erange.includes(x)) {
            std::stringstream ss, ss1;
            ss << x;
            ss1 << trange;
            throw std::out_of_range("Initial element index: " + ss.str() +
                                    " is not in the TiledRange: " + ss1.str());
        }
        const auto t_cardinal_index = trange.element_to_tile(x.m_index);
        const auto tidx = trange.tiles_range().idx(t_cardinal_index);
        new_domain.insert(Index(tidx.begin(), tidx.end()));
    }
    return new_domain;
}

} // namespace tensorwrapper::sparse_map::detail_
