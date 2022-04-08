#pragma once
#include "../../../sparse_map/sparse_map/detail_/tiling_map_index.hpp"
#include "tensorwrapper/tensor/novel/allocators/tiled_array.hpp"
#include "tiled_array_sparse_shape.hpp"
#include "tiled_array_tiling.hpp"

namespace tensorwrapper::tensor::novel::allocator::detail_ {

template<typename ShapeType, typename Op>
default_tensor_type<field::Scalar> generate_ta_scalar_tensor(
  TA::World& world, const ShapeType& shape, ta::Tiling tiling,
  Op&& scalar_fxn) {
    // Get TiledRange for specified tiling
    auto ta_range = make_tiled_range(tiling, shape);

    // Generate the TA tensor
    using tensor_type = default_tensor_type<field::Scalar>;
    using tile_type   = TA::Tensor<double>;
    using range_type  = TA::Range;
    if(scalar_fxn) {
        auto ta_functor = [&](tile_type& t, const range_type& range) {
            const auto lo = range.lobound();
            const auto up = range.upbound();
            sparse_map::Index lo_idx(lo.begin(), lo.end());
            sparse_map::Index up_idx(up.begin(), up.end());
            if(shape.is_hard_zero(lo_idx, up_idx)) {
                return 0.; // Handle manual sparisty
            } else {
                t = tile_type(range, 0.0);    // Create tile;
                scalar_fxn(lo, up, t.data()); // Populate
                return TA::norm(t);           // Handle numerical sparisty
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
  TA::World& world, const ShapeType& shape, ta::Tiling tiling, Op&& tot_fxn) {
    // Get TiledRange for specified tiling
    auto ta_range = make_tiled_range(tiling, shape);

    // Generate the TA tensor
    using tensor_type     = default_tensor_type<field::Tensor>;
    using tile_type       = TA::Tensor<TA::Tensor<double>>;
    using inner_tile_type = TA::Tensor<double>;
    using range_type      = TA::Range;

    std::vector<size_t> inner_lobounds, inner_upbounds;
    {
        inner_upbounds = shape.inner_extents();
        inner_lobounds = std::vector<size_t>(inner_upbounds.size(), 0ul);
    }
    range_type inner_range(inner_lobounds, inner_upbounds);

    if(tot_fxn) {
        auto ta_functor = [=, &tot_fxn, &shape](tile_type& t,
                                                const range_type& range) {
            t = tile_type(range, inner_tile_type(inner_range, 0.0));
            for(auto idx : range) {
                if(!shape.is_hard_zero(sparse_map::Index(idx.begin(), idx.end()))) {
                    std::vector<size_t> outer_index(idx.begin(), idx.end());
                    auto& inner_tile = t[idx];
                    tot_fxn(outer_index, inner_lobounds, inner_upbounds,
                            inner_tile.data());
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

} // namespace tensorwrapper::tensor::novel::allocator::detail_
