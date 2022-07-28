#pragma once
#include <memory>
#include <tiledarray.h>
#include <utility>

// A nearly general TiledArray lazy tile for use in direct methods.
template<typename Tile, typename Builder>
struct LazyTile {
    /// Type of the data tile
    using eval_type = Tile;

    /// The type of the values of the data tile
    using value_type = typename Tile::value_type;

    /// The range of the tile
    TA::Range range;

    /// The builder that produces the tile data on call
    /// Needs to be serializable by madness
    Builder builder;

    /// Normal ctors
    LazyTile()                                 = default;
    LazyTile(const LazyTile& other)            = default;
    LazyTile& operator=(const LazyTile& other) = default;

    /// Ctor that takes range and builder
    LazyTile(TA::Range range, Builder builder) :
      range(range), builder(builder) {}

    /** @brief Convert to data tile type
     *
     *  @returns The filled data tile
     */
    explicit operator eval_type() { return builder(range); }

    /** @brief Serialize the tile
     *
     *  @param ar The archive
     */
    template<typename Archive>
    void serialize(Archive& ar) {
        ar& range;
        ar& builder;
    }

}; // class LazyTile

/** @brief Stream operator for Direct Tile
 *
 *  @param os Stream
 *  @param t The direct tile
 *  @returns Output Stream
 */
template<typename Tile, typename Builder>
std::ostream& operator<<(std::ostream& os, const LazyTile<Tile, Builder>& t) {
    os << t.range << "\n";
    return os;
}