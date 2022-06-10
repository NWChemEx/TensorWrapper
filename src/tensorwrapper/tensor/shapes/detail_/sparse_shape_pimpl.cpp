#include "../../../../sparse_map/sparse_map/detail_/tiling_map_index.hpp"
#include "sparse_shape_pimpl.hpp"

namespace tensorwrapper::tensor::detail_ {

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
//
template<typename FieldType>
bool SPARSE_SHAPE_PIMPL::is_hard_zero(const index_type& el) const {
    const auto nind = m_sm_.ind_rank();
    const auto ndep = m_sm_.dep_rank();
    const auto rank = nind + ndep;

    constexpr bool is_tot = field::is_tensor_field_v<FieldType>;
    const auto max_rank   = is_tot ? nind : rank;
    if(el.size() != max_rank)
        throw std::runtime_error("Slice Rank Inconsistent");

    // Break apart lo/hi into ind/dep indices
    index_type el_ind(el.begin(), el.begin() + nind);

    if constexpr(field::is_scalar_field_v<FieldType>) {
        index_type el_dep(el.begin() + nind, el.end());

        // Find the independent index in sparse map
        if(m_sm_.count(el_ind)) {
            const auto& domain = m_sm_[el_ind];
            return !domain.count(el_dep);
        } else
            return true;

    } else { // ToT
        return !m_sm_.count(el_ind);
    }
}

template<typename FieldType>
bool SPARSE_SHAPE_PIMPL::is_hard_zero(const index_type& lo,
                                      const index_type& hi) const {
    if(lo.size() != hi.size()) throw std::runtime_error("Lo/Hi Inconsistent");

    for(auto i = 0ul; i < lo.size(); ++i) {
        if(lo[i] >= hi[i]) throw std::runtime_error("Lo must be < Hi");
    }

    const auto nind = m_sm_.ind_rank();
    const auto ndep = m_sm_.dep_rank();
    const auto rank = nind + ndep;
    // std::cout << "IND = " << nind << " DEP = " << ndep << std::endl;

    constexpr bool is_tot = field::is_tensor_field_v<FieldType>;
    const auto max_rank   = is_tot ? nind : rank;
    if(lo.size() != max_rank)
        throw std::runtime_error("Slice Rank Inconsistent");

    // Break apart lo/hi into ind/dep indices
    index_type lo_ind(lo.begin(), lo.begin() + nind);
    index_type hi_ind(hi.begin(), hi.begin() + nind);

    if constexpr(field::is_scalar_field_v<FieldType>) {
        index_type lo_dep(lo.begin() + nind, lo.end());
        index_type hi_dep(hi.begin() + nind, hi.end());
        // std::cout << "lo " << lo_ind << ", " << lo_dep << std::endl;
        // std::cout << "hi " << hi_ind << ", " << hi_dep << std::endl;

        // TODO: This is very inefficient, needs to be expressed as a search
        for(auto [ind_idx, domain] : m_sm_) {
#if 0
            // Check if independant index is in the slice
#if 0
            bool ind_in_slice = false;
            for(int i = 0; i < nind; ++i) {
                if(ind_idx[i] >= lo_ind[i] and ind_idx[i] < hi_ind[i]) {
                    ind_in_slice = true;
                    break;
                }
            }
#else
	    auto ind_in_slice = ind_idx >= lo_ind and ind_idx < hi_ind;
#endif
	    std::cout << std::boolalpha;
	    std::cout << "  IND CHECK " << ind_idx << " " << ind_in_slice << std::endl;

            // Check if domain has overlap with slice
            if(ind_in_slice) {
                for(const auto& dep_idx : domain) {
#if 0
                    for(int i = 0; i < ndep; ++i) {
                        if(dep_idx[i] >= lo_dep[i] and dep_idx[i] < hi_dep[i]) {
                            return false;
                        }
                    }
#else
		    std::cout << "    D " << dep_idx << std::endl;
		    if( dep_idx >= lo_dep and dep_idx < hi_dep ) return false;
#endif
                }
            }
#else
            for(const auto& dep_idx : domain) {
                index_type full_idx;
                full_idx.m_index.resize(rank);
                std::copy(ind_idx.begin(), ind_idx.end(), full_idx.begin());
                std::copy(dep_idx.begin(), dep_idx.end(),
                          full_idx.begin() + nind);
                bool point_in_slice = true;
                for(auto i = 0; i < rank; ++i) {
                    point_in_slice =
                      point_in_slice and
                      (full_idx[i] >= lo[i] and full_idx[i] < hi[i]);
                }
                if(point_in_slice) return false;
            }
#endif
        }

    } else { // ToT

        // TODO: This is very inefficient, needs to be expressed as a search
        for(auto [ind_idx, domain] : m_sm_) {
            // Check if independant index is in the slice
            for(int i = 0; i < nind; ++i) {
                if(ind_idx[i] >= lo_ind[i] and ind_idx[i] < hi_ind[i]) {
                    return false;
                }
            }
        }
    }

    return true;
}

template<typename FieldType>
typename SPARSE_SHAPE_PIMPL::pimpl_pointer SPARSE_SHAPE_PIMPL::slice_(
  const index_type& _lo, const index_type& _hi) const {
    // Get base impl (modifies extents, etc)
    auto _base_ptr = base_type::slice_(_lo, _hi);

    // Break bounds into dep/indep components
    index_type lo_ind(_lo.begin(), _lo.begin() + m_sm_.dep_rank());
    index_type hi_ind(_hi.begin(), _hi.begin() + m_sm_.dep_rank());
    index_type lo_dep(_lo.begin() + m_sm_.dep_rank(), _lo.end());
    index_type hi_dep(_hi.begin() + m_sm_.dep_rank(), _hi.end());

    // Creat modified sparse maps
    sparse_map_type new_sm;
    for(const auto& [ind, domain] : m_sm_)
        if(ind >= lo_ind and ind < hi_ind) {
            for(const auto& dep : domain)
                if(dep >= lo_dep and dep < hi_dep) {
                    new_sm.add_to_domain(ind, dep);
                }
        }

    return pimpl_pointer(new my_type(
      _base_ptr->extents(), _base_ptr->inner_extents(), new_sm, m_i2m_));
}

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
