#include "detail_/tiled_sparse_map_pimpl.hpp"
#include "tensorwrapper/sparse_map/sparse_map/sparse_map.hpp"

namespace tensorwrapper::sparse_map {

template<typename DepIndex>
auto* downcast_pimpl(detail_::SparseMapPIMPL<TileIndex, DepIndex>* ppimpl) {
    using base_t  = detail_::SparseMapPIMPL<TileIndex, DepIndex>;
    using pimpl_t = detail_::TiledSparseMapPIMPL<base_t>;
    auto p        = dynamic_cast<pimpl_t*>(ppimpl);
    if(p == nullptr)
        throw std::runtime_error("PIMPL is not a TiledSparseMapPIMPL");
    return p;
}

template<typename DepIndex>
auto* downcast_pimpl(
  const detail_::SparseMapPIMPL<TileIndex, DepIndex>* ppimpl) {
    using base_t  = detail_::SparseMapPIMPL<TileIndex, DepIndex>;
    using pimpl_t = detail_::TiledSparseMapPIMPL<base_t>;
    auto p        = dynamic_cast<const pimpl_t*>(ppimpl);
    if(p == nullptr)
        throw std::runtime_error("PIMPL is not a TiledSparseMapPIMPL");
    return p;
}

inline void check_tr(std::size_t rank, const TA::TiledRange& tr) {
    const bool is_not_set = tr == TA::TiledRange{};
    if(rank > 0 && is_not_set)
        throw std::runtime_error("Input SparseMap's TiledRange is not set");
}

#define SPARSEMAPEE SparseMap<ElementIndex, ElementIndex>
#define SPARSEMAPET SparseMap<ElementIndex, TileIndex>
#define SPARSEMAPTE SparseMap<TileIndex, ElementIndex>
#define SPARSEMAPTT SparseMap<TileIndex, TileIndex>

SPARSEMAPEE::SparseMap(const SPARSEMAPET& other) {
    for(const auto& [oeidx, d] : other) {
        for(const auto& itidx : d) {
            const auto tr = d.trange();
            check_tr(other.dep_rank(), tr);
            for(const auto& ieidx : tr.make_tile_range(itidx)) {
                const ElementIndex temp(ieidx.begin(), ieidx.end());
                add_to_domain(oeidx, temp);
            }
        }
    }
}

SPARSEMAPEE::SparseMap(const SPARSEMAPTE& other) {
    const auto& trange = other.trange();
    check_tr(other.ind_rank(), trange);
    for(const auto& [otidx, d] : other) {
        for(const auto& oeidx : trange.make_tile_range(otidx)) {
            for(const auto& ieidx : d) {
                const ElementIndex temp(oeidx.begin(), oeidx.end());
                add_to_domain(temp, ieidx);
            }
        }
    }
}

SPARSEMAPEE::SparseMap(const SPARSEMAPTT& other) {
    const auto& trange = other.trange();
    check_tr(other.ind_rank(), trange);

    for(const auto& [otidx, d] : other) {
        for(const auto& oeidx : trange.make_tile_range(otidx)) {
            const ElementIndex otemp(oeidx.begin(), oeidx.end());
            for(const auto& itidx : d) {
                check_tr(other.dep_rank(), d.trange());
                for(const auto& ieidx : d.trange().make_tile_range(itidx)) {
                    const ElementIndex itemp(ieidx.begin(), ieidx.end());
                    add_to_domain(otemp, itemp);
                }
            }
        }
    }
}

SPARSEMAPET::SparseMap(const TA::TiledRange& tr, const SPARSEMAPEE& other) {
    for(const auto& [ind, dep] : other) {
        for(auto tidx : TileDomain(tr, dep)) add_to_domain(ind, tidx);
    }
    set_domain_trange(tr);
}

void SPARSEMAPET::set_domain_trange(const TA::TiledRange& tr) {
    for(std::size_t i = 0; i < size(); ++i)
        pimpl_().at(i).second.set_trange(tr);
}

SPARSEMAPTE::SparseMap(const SPARSEMAPTT& other) {
    set_trange(other.trange());
    for(const auto& [otidx, d] : other) {
        const auto& itrange = d.trange();
        check_tr(other.dep_rank(), itrange);
        for(const auto& itidx : d) {
            for(const auto& ieidx : itrange.make_tile_range(itidx)) {
                ElementIndex new_idx(ieidx.begin(), ieidx.end());
                add_to_domain(otidx, std::move(new_idx));
            }
        }
    }
}

SPARSEMAPTE::SparseMap(const TA::TiledRange& trange, const SPARSEMAPEE& other) {
    if(trange.rank() != other.ind_rank())
        throw std::runtime_error("Rank of TiledRange does not equal independent"
                                 " index rank");
    set_trange(trange);
    for(const auto& [oeidx, d] : other) {
        auto otemp = trange.tiles_range().idx(trange.element_to_tile(oeidx));
        TileIndex otidx(otemp.begin(), otemp.end());
        for(const auto& ieidx : d) { add_to_domain(otidx, ieidx); }
    }
}

void SPARSEMAPTE::set_trange(const TA::TiledRange& trange) {
    downcast_pimpl(&pimpl_())->set_trange(trange);
}

const TA::TiledRange& SPARSEMAPTE::trange() const {
    return downcast_pimpl(&pimpl_())->trange();
}

SPARSEMAPTT::SparseMap(const TA::TiledRange& trange, const SPARSEMAPET& other) {
    if(trange.rank() != other.ind_rank())
        throw std::runtime_error("Rank of TiledRange does not equal independent"
                                 " index rank");
    set_trange(trange);
    std::optional<TA::TiledRange> dom_tr;
    for(const auto& [oeidx, d] : other) {
        if(!dom_tr.has_value() && d.trange().rank()) dom_tr = d.trange();
        auto otemp = trange.tiles_range().idx(trange.element_to_tile(oeidx));
        TileIndex otidx(otemp.begin(), otemp.end());
        for(const auto& ieidx : d) { add_to_domain(otidx, ieidx); }
    }
    if(dom_tr.has_value()) set_domain_trange(dom_tr.value());
}

void SPARSEMAPTT::set_trange(const TA::TiledRange& trange) {
    downcast_pimpl(&pimpl_())->set_trange(trange);
}

const TA::TiledRange& SPARSEMAPTT::trange() const {
    return downcast_pimpl(&pimpl_())->trange();
}

void SPARSEMAPTT::set_domain_trange(const TA::TiledRange& tr) {
    for(std::size_t i = 0; i < size(); ++i)
        pimpl_().at(i).second.set_trange(tr);
}

#undef SPARSEMAPEE
#undef SPARSEMAPET
#undef SPARSEMAPTE
#undef SPARSEMAPTT

template class SparseMap<ElementIndex, ElementIndex>;
template class SparseMap<ElementIndex, TileIndex>;
template class SparseMap<TileIndex, ElementIndex>;
template class SparseMap<TileIndex, TileIndex>;

} // namespace tensorwrapper::sparse_map
