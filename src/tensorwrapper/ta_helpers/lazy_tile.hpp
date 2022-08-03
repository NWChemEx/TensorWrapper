#pragma once
#include <memory>
#include <tiledarray.h>
#include <utility>

namespace tensorwrapper::ta_helpers {

// A nearly general TiledArray lazy tile for use in direct methods.
template<typename TileType, typename EvaluatorType>
struct LazyTile {
    /// Type of the data tile
    using eval_type = TileType;

    /// The type of the values of the data tile
    using value_type = typename TileType::value_type;

    /// TiledArray Range type
    using range_type = TA::Range;

    /// The type of the data evaluator
    /// Needs to be serializable by madness
    using evaluator_type = EvaluatorType;

    /// Normal ctors
    LazyTile()                      = default;
    LazyTile(const LazyTile& other) = default;
    LazyTile& operator=(const LazyTile& other) = default;

    /// Ctor that takes range and evaluator
    LazyTile(range_type range, evaluator_type evaluator) :
      m_range_(range), m_evaluator_(evaluator) {}

    /** @brief Convert to data tile type
     *
     *  @returns The filled data tile
     */
    explicit operator eval_type() { return m_evaluator_(m_range_); }

    /** @brief Serialize the tile
     *
     *  @param ar The archive
     */
    template<typename Archive>
    void serialize(Archive& ar) {
        ar& m_range_;
        ar& m_evaluator_;
    }

private:
    /// The range of the tile
    range_type m_range_;

    /// The evaluator that produces the tile data on call
    evaluator_type m_evaluator_;

}; // class LazyTile

/** @brief Stream operator for Direct Tile
 *
 *  @param os Stream
 *  @param t The direct tile
 *  @returns Output Stream
 */
template<typename Tile, typename EvaluatorType>
std::ostream& operator<<(std::ostream& os,
                         const LazyTile<Tile, EvaluatorType>& t) {
    os << t.range << "\n";
    return os;
}

} // namespace tensorwrapper::ta_helpers