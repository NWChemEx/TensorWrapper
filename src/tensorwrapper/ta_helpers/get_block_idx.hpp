#pragma once
#include "ta_headers.hpp"
#include "tensorwrapper/sparse_map/index.hpp"

namespace tensorwrapper::ta_helpers {

/** @brief Gets the tile index associated with an ElementIndex
 *
 *  @param[in] trange The TiledRange the element index is taken from.
 *  @param[in] idx The index of the element.
 *  @return The index of the tile the element belongs to.
 */
inline auto get_block_idx(const TA::TiledRange& trange,
                          const sparse_map::Index& idx) {
    const auto tidx = trange.element_to_tile(idx);
    return trange.tiles_range().idx(tidx);
}

/** @brief Gets the tile index associated with a tile range
 *
 *  @param[in] trange The TiledRange the element index is taken from.
 *  @param[in] range The tile range.
 *  @return An std::vector containing the index of the tile.
 */
inline auto get_block_idx(const TA::TiledRange& trange,
                          const TA::Range& range) {
    const auto tidx = trange.element_to_tile(range.lobound());
    return trange.tiles_range().idx(tidx);
}

/** @brief Gets the tile index associated with a tile range
 *
 *  @tparam TileType Type of the tiles in the distributed array. Assumed to
 *                   satisfy TA's concept of a Tile.
 *  @tparam PolicyType Type of the tensor's sparsity policy.
 *
 *  @param[in] t The DistArray where the tiles are.
 *  @param[in] range The range.
 *  @return An std::vector containing the index of the tile.
 */
template<typename TileType, typename PolicyType>
auto get_block_idx(const TA::DistArray<TileType, PolicyType>& t,
                   const TA::Range& range) {
    return get_block_idx(t.trange(), range);
}

/** @brief Given the actual tile of a tensor determines the block index
 *
 *  This free-function wraps the process of getting a tile's index in the full
 *  tensor given the full tensor and the tile.
 *
 *  @tparam TileType Type of the tiles in the distributed array. Assumed to
 *                   satisfy TA's concept of a Tile.
 *  @tparam PolicyType Type of the tensor's sparsity policy.
 *
 *  @param[in] t The full tensor from which @p tile was taken.
 *  @param[in] tile The tile we want the block index of.
 *
 *  @return An std::vector containing the index of the tile.
 */
template<typename TileType, typename PolicyType>
auto get_block_idx(const TA::DistArray<TileType, PolicyType>& t,
                   const TileType& tile) {
    return get_block_idx(t, tile.range());
}

} // namespace tensorwrapper::ta_helpers
