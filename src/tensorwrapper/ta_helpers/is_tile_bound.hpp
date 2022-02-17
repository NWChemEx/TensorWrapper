#pragma once
#include "get_block_idx.hpp"
#include "ta_headers.hpp"
#include "tensorwrapper/sparse_map/index.hpp"
#include <algorithm> // equal

namespace tensorwrapper::ta_helpers {

using sparse_map::Index;

/** @brief Determines if a particular element index is the first element of a
 *         tile.
 *
 *  This function will consider the tiles in the provided TiledRange and will
 *  return true if one of the tiles has elements in the range `[elem, x)`, where
 *  `elem` is the provided element and `x` is arbitrary other than it is
 *  lexicographically greater than `elem`.
 *
 * @param[in] tr The TiledRange we are searching for @p elem.
 * @param[in] elem The element whose lower-bound-ness is in question.
 *
 * @return True if @p elem is the lower bound of a tile and false otherwise.
 *
 * @throw None No throw guarantee.
 */
inline bool is_tile_lower_bound(const TA::TiledRange& tr,
                                const Index& elem) noexcept {
    TA_ASSERT(elem.size() == tr.rank()); // Doesn't throw, just crashes

    // Make sure the element is actually in the TiledRange
    if(!tr.elements_range().includes(elem)) return false;

    // It is in the range so get the tile it belongs to
    const auto tidx  = get_block_idx(tr, elem);
    const auto& tile = tr.tile(tidx);

    // Now compare to the lower bound of the tile
    const auto lo = tile.lobound();
    return std::equal(lo.begin(), lo.end(), elem.begin());
}

/** @brief Determines if the provided element is an upper bound of any tile in
 *         the range.
 *
 *  This function will consider the tiles in the provided TiledRange and will
 *  return true if one of the tiles has elements in the range `[x, elem)` where
 *  `elem` is the provided element and `x` is arbitrary other than it is
 *  lexicographically less than `elem`.
 *
 *  @note Following usual C++ convention we define the upper bound as "just past
 *        the last element".
 *
 * @param[in] tr The tiled range we are searching for @p elem.
 * @param[in] elem The element whose upper-bound-ness is in question.
 *
 * @return True if @p elem is the upper bound of a tile in @p tr and false
 *         otherwise
 *
 * @throw std::bad_alloc if there insufficient memory to copy @p elem. Strong
 *                       throw guarantee.
 */
inline bool is_tile_upper_bound(const TA::TiledRange& tr, const Index& elem) {
    using size_type = typename Index::value_type;

    // TODO: Make no throw guarantee by taking a copy of elem and manipulating
    //       the copy.

    TA_ASSERT(elem.size() == tr.rank());

    // If this index is an upper bound, the last element in the tile it would be
    // an upper bound of would have offsets along mode i equal to elem[i] - 1
    std::vector<size_type> elem_m1(elem.begin(), elem.end());
    for(auto& x : elem_m1) {
        if(x == 0) return false; // Tiles have to have elements
        --x;
    }

    // Make sure that elem - 1 is in the tiled range
    if(!tr.elements_range().includes(elem_m1)) return false;

    const auto tidx  = get_block_idx(tr, Index(elem_m1));
    const auto& tile = tr.tile(tidx);
    const auto& hi   = tile.upbound();
    return std::equal(hi.begin(), hi.end(), elem.begin());
}

} // namespace tensorwrapper::ta_helpers
