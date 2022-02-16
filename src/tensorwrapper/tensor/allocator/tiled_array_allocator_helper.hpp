#pragma once
#include "../../sparse_map/sparse_map/detail_/tiling_map_index.hpp"
#include "tensorwrapper/tensor/allocators/tiled_array.hpp"
#include "tiled_array_sparse_shape.hpp"
#include "tiled_array_tiling.hpp"

namespace tensorwrapper::tensor::allocator::detail_ {

template<typename ShapeType, typename Op>
default_tensor_type<field::Scalar> generate_ta_scalar_tensor(
  TA::World& world, const ShapeType& shape, ta::Tiling tiling,
  Op&& scalar_fxn) {
    // Get TiledRange for specified tiling
    auto ta_range = make_tiled_range(tiling, shape);

    // TODO Handle possible sparse_map driven shape
    // auto ta_shape = make_sparse_shape<field::Scalar>(shape, ta_trange);

    // Generate the TA tensor
    using tensor_type = default_tensor_type<field::Scalar>;
    using tile_type   = TA::Tensor<double>;
    using range_type  = TA::Range;
    if(scalar_fxn) {
        auto ta_functor = [&](tile_type& t, const range_type& range) {
            t = tile_type(range, 0.0); // Create tile;
            scalar_fxn(range.lobound(), range.upbound(), t.data()); // Populate
            return TA::norm(t); // Handle numerical sparisty
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

    // TODO Handle possible sparse_map driven shape
    // auto ta_shape = make_sparse_shape<field::Scalar>(shape, ta_trange);

    // Generate the TA tensor
    using tensor_type = default_tensor_type<field::Tensor>;
    using tile_type   = TA::Tensor<TA::Tensor<double>>;
    using range_type  = TA::Range;
    if(tot_fxn) {
        auto ta_functor = [&](tile_type& t, const range_type& range) {
            std::cout << range << std::endl;
            return 1.;
        };
        return TA::make_array<tensor_type>(world, ta_range, ta_functor);
    } else {
        throw std::runtime_error("Must Specify Valid Population Fxn");
        return tensor_type();
    }
}

} // namespace tensorwrapper::tensor::allocator::detail_
