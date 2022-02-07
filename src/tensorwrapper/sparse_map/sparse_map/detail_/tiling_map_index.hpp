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
sparse_map::SparseMap tile_dependent_indices(const sparse_map::SparseMap& sm,
                                             const TA::TiledRange& tr) {
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
sparse_map::SparseMap tile_indices(const sparse_map::SparseMap& sm,
                                   const TA::TiledRange& ind_tr,
                                   const TA::TiledRange& dep_tr, ) {
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
        for(const auto& new_idx : tr.make_tile_range(ind_idx)) {
            for(const auto& dep_idx : d) {
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
            for(const auto& new_idx : tr.make_tile_range(dep_idx)) {
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
sparse_map::SparseMap untile_indices(const sparse_map::SparseMap& sm,
                                     const TA::TiledRange& ind_tr,
                                     const TA::TiledRange& dep_tr, ) {
    auto intermediate_sm = untile_independent_indices(sm, ind_tr);
    return untile_dependent_indices(intermediate_sm, dep_tr);
}

} // namespace tensorwrapper::sparse_map::detail_
