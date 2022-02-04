#include "sparse_map_pimpl.hpp"
#include "tensorwrapper/sparse_map/sparse_map/sparse_map.hpp"

namespace tensorwrapper::sparse_map {

//------------------------------------------------------------------------------
//                            CTors
//------------------------------------------------------------------------------

SparseMap::SparseMap() : m_pimpl_(std::make_unique<pimpl_type>()) {}

SparseMap::SparseMap(il_t il) : SparseMap() {
    for(auto&& [k, v] : il)
        for(const auto& dep : v) add_to_domain(k, dep);
}

SparseMap::SparseMap(const SparseMap& rhs) :
  m_pimpl_(std::make_unique<pimpl_type>(rhs.pimpl_())) {}

SparseMap::SparseMap(SparseMap&&) noexcept = default;

SparseMap& SparseMap::operator=(const SparseMap& rhs) {
    if(this == &rhs) return *this;
    m_pimpl_ = std::make_unique<pimpl_type>(rhs.pimpl_());
    return *this;
}

SparseMap& SparseMap::operator=(SparseMap&&) noexcept = default;

SparseMap::~SparseMap() noexcept = default;

//------------------------------------------------------------------------------
//                                 Accessors
//------------------------------------------------------------------------------

typename SparseMap::size_type SparseMap::size() const noexcept {
    return m_pimpl_ ? pimpl_().size() : 0;
}

bool SparseMap::count(const key_type& i) const noexcept {
    if(m_pimpl_)
        return pimpl_().count(i);
    else
        return false;
}

typename SparseMap::size_type SparseMap::ind_rank() const noexcept {
    if(!m_pimpl_)
        return 0;
    else
        return pimpl_().ind_rank();
}

typename SparseMap::size_type SparseMap::dep_rank() const noexcept {
    if(!m_pimpl_)
        return 0;
    else
        return pimpl_().dep_rank();
}

void SparseMap::add_to_domain(const key_type& key, Index value) {
    if(!m_pimpl_) m_pimpl_ = std::make_unique<pimpl_type>();
    pimpl_().add_to_domain(key, std::move(value));
}

const typename SparseMap::value_type& SparseMap::operator[](size_type i) const {
    return pimpl_().at(i);
}

const typename SparseMap::mapped_type& SparseMap::operator[](
  const key_type& key) const {
    if(!m_pimpl_) {
        std::stringstream ss;
        ss << "Index: " << key << " is not in range [0, 0)";
        throw std::out_of_range(ss.str());
    }
    return pimpl_().at(key);
}

SparseMap SparseMap::direct_product(const SparseMap& rhs) const {
    if(!m_pimpl_ || empty()) return SparseMap{};
    if(!rhs.m_pimpl_ || rhs.empty()) { return SparseMap{}; }
    SparseMap rv(*this);
    rv.pimpl_().direct_product_assign(rhs.pimpl_());
    return rv;
}

SparseMap SparseMap::operator*(const SparseMap& rhs) const {
    SparseMap rv(*this);
    rv *= rhs;
    return rv;
}

SparseMap& SparseMap::operator*=(const SparseMap& rhs) {
    if(!m_pimpl_ || !rhs.m_pimpl_) {
        m_pimpl_ = new_pimpl<IndIndex, DepIndex>();
        return *this;
    }
    pimpl_() *= rhs.pimpl_();
    return *this;
}

SparseMap& SparseMap::operator^=(const SparseMap& rhs) {
    if(!m_pimpl_ || !rhs.m_pimpl_) {
        m_pimpl_ = new_pimpl<IndIndex, DepIndex>();
        return *this;
    }
    pimpl_() ^= rhs.pimpl_();
    return *this;
}

SparseMap SparseMap::operator^(const SparseMap& rhs) const {
    SparseMap rv(*this);
    rv ^= rhs;
    return rv;
}

SparseMap SparseMap::inverse() const {
    SparseMap rv;
    if(empty()) return rv;

    for(const auto& [ind, domain] : *this) {
        for(const auto& dep : domain) rv.add_to_domain(dep, ind);
    }
    return rv;
}

SparseMap& SparseMap::operator+=(const SparseMap& rhs) {
    if(!m_pimpl_) {
        if(!rhs.m_pimpl_)
            m_pimpl_ = new_pimpl<IndIndex, DepIndex>();
        else
            m_pimpl_ = std::make_unique<pimpl_type>(rhs.pimpl_());
        return *this;
    } else if(!rhs.m_pimpl_)
        return *this;
    pimpl_() += rhs.pimpl_();
    return *this;
}

SparseMap SparseMap::operator+(const SparseMap& rhs) const {
    auto rv = SparseMap(*this);
    rv += rhs;
    return rv;
}

bool SparseMap::operator==(const SparseMap& rhs) const noexcept {
    if(!m_pimpl_)
        return !rhs.m_pimpl_ || rhs.empty();
    else if(!rhs.m_pimpl_)
        return empty();
    return pimpl_() == rhs.pimpl_();
}

void SparseMap::hash(tensorwrapper::detail_::Hasher& h) const {
    if(m_pimpl_)
        pimpl_().hash(h);
    else
        h(nullptr);
}

std::ostream& SparseMap::print(std::ostream& os) const {
    os << pimpl_();
    return os;
}

typename SparseMap::pimpl_type& SparseMap::pimpl_() {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error("PIMPL not set. Did you move from this instance?");
}

const typename SparseMap::pimpl_type& SparseMap::pimpl_() const {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error("PIMPL not set. Did you move from this instance?");
}

//------------------------------------------------------------------------------
//                            Private Methods
//------------------------------------------------------------------------------

SparseMap SparseMap::chain(const SparseMap& sm) const {
    if(dep_rank() != sm.ind_rank())
        throw std::runtime_error(
          "Incompatible index ranks between chained maps");
    SparseMap rv;
    for(const auto& [lind, ldom] : *this) {
        for(const auto& ldep : ldom) {
            if(sm.count(ldep)) {
                for(const auto& rdep : sm.at(ldep)) {
                    rv.add_to_domain(lind, rdep);
                }
            }
        }
    }
    return rv;
}

} // namespace tensorwrapper::sparse_map
