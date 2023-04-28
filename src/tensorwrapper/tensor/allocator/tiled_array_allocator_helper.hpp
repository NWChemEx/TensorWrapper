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
#include "../../sparse_map/sparse_map/detail_/tiling_map_index.hpp"
#include "../../ta_helpers/lazy_tile.hpp"
#include "tensorwrapper/tensor/allocator/direct_tiled_array.hpp"
#include "tensorwrapper/tensor/allocator/tiled_array.hpp"
#include "tiled_array_sparse_shape.hpp"
#include "tiled_array_tiling.hpp"
#include <TiledArray/conversions/make_array.h>

namespace tensorwrapper::tensor::allocator::detail_ {

template<typename Op>
struct is_scalar_tile_fxn {
    static constexpr bool value =
      std::is_invocable_v<Op, std::vector<size_t>, std::vector<size_t>,
                          double*>;
};
template<typename Op>
struct is_tot_tile_fxn {
    static constexpr bool value =
      std::is_invocable_v<Op, std::vector<size_t>, std::vector<size_t>,
                          std::vector<size_t>, double*>;
};

template<typename Op>
inline constexpr bool is_scalar_tile_fxn_v = is_scalar_tile_fxn<Op>::value;
template<typename Op>
inline constexpr bool is_tot_tile_fxn_v = is_tot_tile_fxn<Op>::value;

template<typename ShapeType, typename Op>
default_tensor_type<field::Scalar> generate_ta_scalar_tensor(
  TA::World& world, const ShapeType& shape, Op&& scalar_fxn) {
    // Get TiledRange for specified tiling
    auto ta_range = make_tiled_range(shape);

    // Generate the TA tensor
    using tensor_type = default_tensor_type<field::Scalar>;
    using tile_type   = TA::Tensor<double>;
    using range_type  = TA::Range;
    if(scalar_fxn) {
        auto ta_functor = [=](tile_type& t, const range_type& range) {
            const auto lo = range.lobound();
            const auto up = range.upbound();
            sparse_map::Index lo_idx(lo.begin(), lo.end());
            sparse_map::Index up_idx(up.begin(), up.end());
            if(shape.is_hard_zero(lo_idx, up_idx)) {
                return 0.; // Handle manual sparisty
            } else {
                t = tile_type(range, 0.0); // Create tile;
                // Populate
                if constexpr(is_scalar_tile_fxn_v<Op>) {
                    scalar_fxn(lo, up, t.data());
                } else {
                    for(const auto& idx : range) {
                        std::vector<size_t> _idx(idx.begin(), idx.end());
                        t[idx] = scalar_fxn(_idx);
                    }
                }
                return TA::norm(t); // Handle numerical sparisty
            }
        };
        return TA::make_array<tensor_type>(world, ta_range, ta_functor);
    } else {
        throw std::runtime_error("Must Specify Valid Population Fxn");
        return tensor_type();
    }
}

template<typename ShapeType, typename Op>
default_tensor_type<field::Tensor> generate_ta_tot_tensor(
  TA::World& world, const ShapeType& shape, Op&& tot_fxn) {
    // Get TiledRange for specified tiling
    auto ta_range = make_tiled_range(shape);

    // Generate the TA tensor
    using tensor_type     = default_tensor_type<field::Tensor>;
    using tile_type       = TA::Tensor<TA::Tensor<double>>;
    using inner_tile_type = TA::Tensor<double>;
    using range_type      = TA::Range;

    if(tot_fxn) {
        auto ta_functor = [=](tile_type& t,
                                                const range_type& range) {
            t = tile_type(range);
            for(auto oidx : range) {
                auto nwx_outer_idx =
                  sparse_map::Index(oidx.begin(), oidx.end());
                if(!shape.is_hard_zero(nwx_outer_idx)) {
                    std::vector<size_t> outer_index(oidx.begin(), oidx.end());
                    auto& inner_tile = t[oidx];

                    // Get inner tile dimension
                    const auto& inner_extents =
                      shape.inner_extents().at(nwx_outer_idx).extents();
                    std::vector<size_t> up_bound(inner_extents.begin(),
                                                 inner_extents.end());
                    std::vector<size_t> lo_bound(inner_extents.size(), 0);
                    range_type inner_range(lo_bound, up_bound);

                    // Create Tile
                    inner_tile = inner_tile_type(inner_range, 0.);

                    if constexpr(is_tot_tile_fxn_v<Op>) {
                        tot_fxn(outer_index, lo_bound, up_bound,
                                inner_tile.data());
                    } else {
                        for(const auto& iidx : inner_range) {
                            std::vector<size_t> _idx(iidx.begin(), iidx.end());
                            inner_tile[iidx] = tot_fxn(outer_index, _idx);
                        }
                    }
                }
            }

            return 1.; // XXX: Need to devise a consistent norm here
        };
        return TA::make_array<tensor_type>(world, ta_range, ta_functor);
    } else {
        throw std::runtime_error("Must Specify Valid Population Fxn");
        return tensor_type();
    }
}

template<typename ShapeType, typename Op>
lazy_tensor_type<field::Scalar> generate_ta_scalar_direct_tensor(
  TA::World& world, const ShapeType& shape, std::string fxn_id,
  Op&& scalar_fxn) {
    // Get TiledRange for tiling
    auto ta_range = make_tiled_range(shape);

    // Generate the TA tensor
    using tensor_type = lazy_tensor_type<field::Scalar>;
    using lazy_type   = tensorwrapper::ta_helpers::lazy_scalar_type;
    using tile_type   = typename lazy_type::eval_type;
    using range_type  = TA::Range;
    if(scalar_fxn) {
        /// Wrap scalar_fxn in tile_evaluator and register with fxn_id
        auto tile_evaluator = [=](range_type range) {
            /// Create tile
            tile_type t(range, 0.0);
            /// Check if tile is zero
            const auto lo = range.lobound();
            const auto up = range.upbound();
            sparse_map::Index lo_idx(lo.begin(), lo.end());
            sparse_map::Index up_idx(up.begin(), up.end());
            if(!shape.is_hard_zero(lo_idx, up_idx)) {
                // Populate
                if constexpr(is_scalar_tile_fxn_v<Op>) {
                    scalar_fxn(lo, up, t.data());
                } else {
                    for(const auto& idx : range) {
                        std::vector<size_t> _idx(idx.begin(), idx.end());
                        t[idx] = scalar_fxn(_idx);
                    }
                }
            }
            return t;
        };
        lazy_type::add_evaluator(tile_evaluator, fxn_id);

        /// Make lazy tiles
        auto ta_functor = [fxn_id](lazy_type& t, const range_type& r) -> float {
            t = lazy_type(r, fxn_id);
            return 1.0;
        };
        return TA::make_array<tensor_type>(world, ta_range, ta_functor);
    } else {
        throw std::runtime_error("Must Specify Valid Population Fxn");
        return tensor_type();
    }
}

template<typename ShapeType, typename Op>
lazy_tensor_type<field::Tensor> generate_ta_tot_direct_tensor(
  TA::World& world, const ShapeType& shape, std::string fxn_id, Op&& tot_fxn) {
    // Get TiledRange for tiling
    auto ta_range = make_tiled_range(shape);

    // Generate the TA tensor
    using tensor_type     = lazy_tensor_type<field::Tensor>;
    using lazy_type       = tensorwrapper::ta_helpers::lazy_tot_type;
    using tile_type       = typename lazy_type::eval_type;
    using inner_tile_type = TA::Tensor<double>;
    using range_type      = TA::Range;

    if(tot_fxn) {
        /// Wrap tot_fxn in tile_evaluator and register with fxn_id
        auto tile_evaluator = [=](range_type range) {
            tile_type t(range);
            for(auto idx : range) {
                auto nwx_outer_idx = sparse_map::Index(idx.begin(), idx.end());
                if(!shape.is_hard_zero(nwx_outer_idx)) {
                    std::vector<size_t> outer_index(idx.begin(), idx.end());
                    auto& inner_tile = t[idx];

                    // Get inner tile dimension
                    const auto& inner_extents =
                      shape.inner_extents().at(nwx_outer_idx).extents();
                    std::vector<size_t> up_bound(inner_extents.begin(),
                                                 inner_extents.end());
                    std::vector<size_t> lo_bound(inner_extents.size(), 0);
                    range_type inner_range(lo_bound, up_bound);

                    // Create Tile
                    inner_tile = inner_tile_type(inner_range, 0.);

                    if constexpr(is_tot_tile_fxn_v<Op>) {
                        tot_fxn(outer_index, lo_bound, up_bound,
                                inner_tile.data());
                    } else {
                        for(const auto& idx : inner_range) {
                            std::vector<size_t> _idx(idx.begin(), idx.end());
                            inner_tile[idx] = tot_fxn(outer_index, _idx);
                        }
                    }
                }
            }
            return t;
        };
        lazy_type::add_evaluator(tile_evaluator, fxn_id);

        /// Make lazy tiles
        auto ta_functor = [fxn_id](lazy_type& t, const range_type& r) -> float {
            t = lazy_type(r, fxn_id);
            return 1.0;
        };
        return TA::make_array<tensor_type>(world, ta_range, ta_functor);
    } else {
        throw std::runtime_error("Must Specify Valid Population Fxn");
        return tensor_type();
    }
}

} // namespace tensorwrapper::tensor::allocator::detail_
