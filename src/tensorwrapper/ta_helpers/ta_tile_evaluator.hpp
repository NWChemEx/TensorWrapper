#pragma once
#include <tensorwrapper/tensor/data_evaluator/element_evaluator.hpp>
#include <tensorwrapper/tensor/data_evaluator/tile_evaluator.hpp>
#include <tensorwrapper/tensor/fields.hpp>
#include <tiledarray.h>
#include <variant>

namespace tensorwrapper::ta_helpers {

template<typename TileType>
class TATileEvaluator {
    /// The type this evaluates to
    using tile_t = TileType;

    /// TA Range
    using range_t = TA::Range;

    /// Field associated with TileType
    using field_t = tensor::field::Scalar;

    /// The type of a tile based evaluator
    using tile_evaluator_t = tensor::data_evaluator::TileEvaluator<field_t>;

    /// The type of an element based evaluator
    using element_evaluator_t =
      tensor::data_evaluator::ElementEvaluator<field_t>;

    /// The type of the stored evaluator
    using variant_t = std::variant<tile_evaluator_t, element_evaluator_t>;

    /// Holds the internal data evaluator
    variant_t m_evaluator_;

public:
    TATileEvaluator(tile_evaluator_t& te) : m_evaluator_(te){};
    TATileEvaluator(element_evaluator_t& ee) : m_evaluator_(ee){};

    tile_t operator()(range_t range) {
        auto eval = [=](auto&& evaluator) {
            return evaluate_(range, evaluator);
        };
        return std::visit(eval, m_evaluator_);
    }

    /** @brief Serialize this evaluator
     *
     *  @param ar The archive
     */
    template<typename Archive>
    void serialize(Archive& ar) {
        /// TODO: Doubt this will work, might have to visit on it
        ar& m_evaluator_;
    }

private:
    tile_t evaluate_(range_t& range, tile_evaluator_t& e) {
        auto t = tile_t(range, 0.0);
        e(range.lobound(), range.upbound(), t.data());
    }

    tile_t evaluate_(range_t& range, element_evaluator_t& e) {
        auto t = tile_t(range, 0.0);
        for(const auto& idx : range) {
            std::vector<size_t> _idx(idx.begin(), idx.end());
            t[idx] = e(_idx);
        }
    }

}; /// class TATileEvaluator

} // namespace tensorwrapper::ta_helpers