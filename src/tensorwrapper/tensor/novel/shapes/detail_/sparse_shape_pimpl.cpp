#include "../../../../sparse_map/sparse_map/detail_/tiling_map_index.hpp"
#include "sparse_shape_pimpl.hpp"

namespace tensorwrapper::tensor::novel::detail_ {

#define SPARSE_SHAPE_PIMPL SparseShapePIMPL<FieldType>

// These are the same for either field
//{
using sm_type       = SparseShapePIMPL<field::Scalar>::sparse_map_type;
using idx2mode_type = SparseShapePIMPL<field::Scalar>::idx2mode_type;
using ta_shape_type = SparseShapePIMPL<field::Scalar>::ta_shape_type;
using ta_tile_range = SparseShapePIMPL<field::Scalar>::ta_tile_range;
//}

using size_type  = unsigned int;
using index_type = std::vector<size_type>;
using tile_index = tensorwrapper::sparse_map::Index;

namespace {

// Makes a tiled range for the provided slice
auto make_tr(const idx2mode_type& idx2mode, const ta_tile_range& tr) {
    using tr1_type     = TA::TiledRange1;
    using tr1_vec_type = std::vector<tr1_type>;
    using size_type    = typename tr1_vec_type::size_type;

    const auto nidxs = idx2mode.size();
    tr1_vec_type tr1s(nidxs);
    for(size_type i = 0; i < nidxs; ++i) tr1s[i] = tr.dim(idx2mode[i]);

    return ta_tile_range(tr1s.begin(), tr1s.end());
}

auto sm_to_tensor_shape(const sm_type& sm, const idx2mode_type& i2m,
                        const ta_tile_range& tr) {
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
    const auto ind_tr = make_tr(ind, tr);
    const auto dep_tr = make_tr(dep, tr);

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

// Similar to tensor implementation, but only need shape for independent indices
auto sm_to_tot_shape(const sm_type& sm, const idx2mode_type& i2m,
                     const ta_tile_range& tr) {
    const auto nind = sm.ind_rank();

    if(nind != tr.rank())
        throw std::runtime_error("SparseMap not consistent with TiledRange");

    using float_type = float;
    TA::Tensor<float_type> shape_data(tr.tiles_range(), 0.0);
    auto smTE =
      tensorwrapper::sparse_map::detail_::tile_independent_indices(sm, tr);

    index_type full_index(nind);
    for(const auto& [ind_idx, _] : smTE) {
        for(size_type i = 0; i < nind; ++i) full_index[i2m[i]] = ind_idx[i];
        shape_data[full_index] = std::numeric_limits<float_type>::max();
    }
    return ta_shape_type(shape_data, tr);
}

} // namespace

template<typename FieldType>
SPARSE_SHAPE_PIMPL::SparseShapePIMPL(extents_type x, inner_extents_type y,
		                     sparse_map_type sm, idx2mode_type i2m) :
  m_sm_(std::move(sm)),
  m_i2m_(std::move(i2m)),
  base_type(std::move(x), std::move(y)) {
    const auto nind = m_sm_.ind_rank();
    const auto ndep = m_sm_.dep_rank();
    const auto rank = nind + ndep;

    constexpr bool is_tot = field::is_tensor_field_v<FieldType>;
    const auto max_rank   = is_tot ? nind : rank;

    if(max_rank != this->extents().size())
        throw std::runtime_error("Rank of SparseMap is not consistent with the "
                                 "provided extents");

    if(max_rank != m_i2m_.size())
        throw std::runtime_error("SparseMap not consistent with idx2mode");

    for(const auto x : m_i2m_)
        if(x >= max_rank)
            throw std::out_of_range("Index maps to mode outside range [0, " +
                                    std::to_string(max_rank) + ")");
}

template<typename FieldType>
typename SPARSE_SHAPE_PIMPL::ta_shape_type SPARSE_SHAPE_PIMPL::shape(
  const ta_tile_range& tr) const {
    if constexpr(field::is_scalar_field_v<FieldType>) {
        return sm_to_tensor_shape(m_sm_, m_i2m_, tr);
    } else {
        return sm_to_tot_shape(m_sm_, m_i2m_, tr);
    }
}

template<typename FieldType>
bool SPARSE_SHAPE_PIMPL::operator==(
  const SparseShapePIMPL& rhs) const noexcept {
    if(std::tie(m_sm_, m_i2m_) == std::tie(rhs.m_sm_, rhs.m_i2m_))
        return base_type::operator==(rhs);
    return false;
}

//------------------------------------------------------------------------------
//                    Protected/Private Member Functions
//------------------------------------------------------------------------------

template<typename FieldType>
typename SPARSE_SHAPE_PIMPL::pimpl_pointer SPARSE_SHAPE_PIMPL::clone_() const {
    return pimpl_pointer(new my_type(*this));
}

template<typename FieldType>
void SPARSE_SHAPE_PIMPL::hash_(tensorwrapper::detail_::Hasher& h) const {
    h(m_sm_, m_i2m_);
    base_type::hash_(h);
}

#undef SPARSE_SHAPE_PIMPL

template class SparseShapePIMPL<field::Scalar>;
template class SparseShapePIMPL<field::Tensor>;

} // namespace tensorwrapper::tensor::detail_
