#pragma once
#include "tiled_array_types.hpp"

namespace tensorwrapper::tensor::allocator::detail_ {

/**
 *  Creates a field-generic TA::TiledRange instance for the OneBigTile tiling
 *
 *  @tparam ShapeType Strong type correponding to shape instance (Field generic)
 *
 *  @param[in] shape The shape of the tensor for which to generate the tiling
 *  @returns   TA::TiledRange corresponding to `shape` in OneBigTile concept
 */
template <typename ShapeType>
tiled_range_type make_one_big_tile_tiled_range( const ShapeType& shape ) {
    using size_type = typename ShapeType::size_type;

    const auto extents = shape.extents();
    const auto n_modes = extents.size();
    std::vector<tr1_type> tr1s(n_modes);
    for(size_type i = 0; i < n_modes; ++i ) {
        tr1s[i] = tr1_type{0, extents[i]};
    }
    return tiled_range_type( tr1s.begin(), tr1s.end() );
}


/**
 *  Creates a field-generic TA::TiledRange for a specific tiling scheme
 *
 *  @tparam ShapeType Strong type correponding to shape instance (Field generic)
 *
 *  @param[in] tiling The tiling scheme from which to generate the tiling range
 *  @param[in] shape The shape of the tensor for which to generate the tiling
 *  @returns   TA::TiledRange corresponding to `shape` in the `tiling` scheme
 */
template <typename ShapeType>
tiled_range_type make_tiled_range( ta::Tiling tiling, const ShapeType& shape ) {
    switch(tiling) {
        case ta::Tiling::OneBigTile:
            return make_one_big_tile_tiled_range(shape);
	default:
	    throw std::runtime_error("Provided Tiling Not Supported");
    }
    return tiled_range_type();
}


}
