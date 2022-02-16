#pragma once
#include "tiled_array_types.hpp"

namespace tensorwrapper::tensor::allocator::detail_ {

template<typename T, typename U>
T downcast(U* ptr) {
    auto dc_ptr = dynamic_cast<T>(ptr);
    if(!ptr) throw std::bad_cast();
    return dc_ptr;
}

// Makes a tiled range for the provided slice
inline auto make_tiled_range(const idx2mode_type& idx2mode,
                             const tiled_range_type& tr) {
    using tr1_vec_type = std::vector<tr1_type>;
    using size_type    = typename tr1_vec_type::size_type;

    const auto nidxs = idx2mode.size();
    tr1_vec_type tr1s(nidxs);
    for(size_type i = 0; i < nidxs; ++i) tr1s[i] = tr.dim(idx2mode[i]);

    return tiled_range_type(tr1s.begin(), tr1s.end());
}

inline auto scalar_tensor_shape(const sparse_map_type& sm,
                                const idx2mode_type& i2m,
                                const tiled_range_type& tr) {
    const auto nind = sm.ind_rank();
    const auto ndep = sm.dep_rank();
    const auto rank = nind + ndep;

    if(rank != tr.rank())
        throw std::runtime_error("SparseMap not consistent with TiledRange");

    // Iterator just past last independent index
    auto ind_end = i2m.begin() + nind;

    // Break the idx2mode up into ind and dep pieces
    idx2mode_type ind(i2m.begin(), ind_end);
    idx2mode_type dep(ind_end, i2m.end());

    // Get the TRs for each piece
    const auto ind_tr = make_tiled_range(ind, tr);
    const auto dep_tr = make_tiled_range(dep, tr);

    // Make a tile-to-tile SM
    auto smTT =
      tensorwrapper::sparse_map::detail_::tile_indices(sm, ind_tr, dep_tr);

    using float_type = float;

    TA::Tensor<float_type> shape_data(tr.tiles_range(), 0.0);
    index_type full_idx(rank);
    for(const auto& [ind_idx, domain] : smTT) {
        for(size_type i = 0; i < nind; ++i) full_idx[ind[i]] = ind_idx[i];
        for(const auto& dep_idx : domain) {
            for(size_type i = 0; i < ndep; ++i) full_idx[dep[i]] = dep_idx[i];
            shape_data[full_idx] = std::numeric_limits<float_type>::max();
        }
    }
    return ta_shape_type(shape_data, tr);
}

template<typename FieldType, typename ShapeType>
ta_shape_type make_sparse_shape(const ShapeType& shape,
                                const tiled_range_type& tiled_range) {
    // Attempt to downcast to SparseShape
    auto sparse_shape_ptr = downcast<const SparseShape<FieldType>*>(&shape);

    auto sparse_map   = sparse_shape_ptr->sparse_map();
    auto idx2mode_map = sparse_shape_ptr->idx2mode_map();

    return scalar_tensor_shape(sparse_map, idx2mode_map, tiled_range);
}

} // namespace tensorwrapper::tensor::allocator::detail_
