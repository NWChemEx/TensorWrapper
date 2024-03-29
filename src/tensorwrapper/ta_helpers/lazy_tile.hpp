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
#include <memory>
#include <tiledarray.h>
#include <utility>

namespace tensorwrapper::ta_helpers {

template<typename TileType>
struct LazyTile {
    using my_type = LazyTile<TileType>;

    /// Type of the data tile
    using eval_type = TileType;

    /// The type of the values of the data tile
    using value_type = typename TileType::value_type;

    /// The scalar type of the data tile
    using scalar_type = typename TileType::scalar_type;

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
    explicit operator eval_type();

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
    static void add_evaluator(evaluator_type evaluator, id_type id);

    /** @brief Prints the tile
     *
     *  @param[in, out] os The stream to print this Tile to. After the
     *                     call a string representation of this instance
     * will have been added to @p os.
     *  @return @p os is returned to support chaining.
     */
    std::ostream& print(std::ostream& os) const {
        os << m_range_;
        return os;
    }

    /** @brief Clones this tile.
     */
    my_type clone() const { return my_type{m_range_, m_id_}; }

private:
    /// The range of the tile
    range_type m_range_;

    /// The evaluator id of this tile
    id_type m_id_;

    /// The map holding the evaluators
    static map_type evaluators;

}; // class LazyTile

/** @brief Stream operator for Direct Tile
 *
 *  @param os Stream
 *  @param t The direct tile
 *  @returns Output Stream
 */
template<typename Tile>
std::ostream& operator<<(std::ostream& os, const LazyTile<Tile>& t) {
    return t.print(os);
}

/// Useful typedefs
using lazy_scalar_type = LazyTile<TA::Tensor<double>>;
using lazy_tot_type    = LazyTile<TA::Tensor<TA::Tensor<double>>>;

} // namespace tensorwrapper::ta_helpers
