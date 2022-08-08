#pragma once
#include <memory>
#include <tiledarray.h>
#include <utility>

namespace tensorwrapper::ta_helpers {

template<typename TileType>
struct LazyTile {
    /// Type of the data tile
    using eval_type = TileType;

    /// The type of the values of the data tile
    using value_type = typename TileType::value_type;

    /// TiledArray Range type
    using range_type = TA::Range;

    /// The type of the data evaluator
    using evaluator_type = std::function<eval_type(range_type)>;

    /// The type of the evaluator id
    using id_type = std::string;

    /// The type of the map of ids to evaluators
    using map_type = std::map<id_type, evaluator_type>;

    /// Normal ctors
    LazyTile()                      = default;
    LazyTile(const LazyTile& other) = default;
    LazyTile& operator=(const LazyTile& other) = default;

    /** @brief Adds an evaluator into the map with a given id.
     *
     *  @param range The range of this tile
     *  @param id The id to this tile's evaluator
     */
    LazyTile(range_type range, id_type id) : m_range_(range), m_id_(id) {}

    /** @brief Convert to data tile type
     *
     *  @returns The filled data tile
     */
    explicit operator eval_type() { return evaluators[m_id_](m_range_); }

    /** @brief Serialize the tile
     *
     *  @param ar The archive
     */
    template<typename Archive>
    void serialize(Archive& ar) {
        ar& m_range_;
        ar& m_id_;
    }

    /** @brief Adds an evaluator into the map with a given id.
     *
     *  @param evaluator A callable evaluator
     *  @param id The id that will be associated with the evaluator
     */
    static void add_evaluator(evaluator_type evaluator, id_type id) {
        if(!evaluators.count(id)) evaluators[id] = evaluator;
    }

private:
    /// The range of the tile
    range_type m_range_;

    /// The evaluator id of this tile
    id_type m_id_;

    /// The map holding the evaluators
    static map_type evaluators;

}; // class LazyTile

/// Useful typedefs
using lazy_scalar_type = LazyTile<TA::Tensor<double>>;
using lazy_tot_type    = LazyTile<TA::Tensor<TA::Tensor<double>>>;

extern template class LazyTile<TA::Tensor<double>>;
extern template class LazyTile<TA::Tensor<TA::Tensor<double>>>;

} // namespace tensorwrapper::ta_helpers