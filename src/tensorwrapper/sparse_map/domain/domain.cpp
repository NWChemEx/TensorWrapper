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

#include "detail_/domain_pimpl.hpp"
#include "tensorwrapper/sparse_map/domain/domain.hpp"
#include <utilities/printing/print_stl.hpp>

namespace tensorwrapper::sparse_map {

//------------------------------------------------------------------------------
//                               CTors
//------------------------------------------------------------------------------

Domain::Domain() : m_pimpl_(std::make_unique<pimpl_type>()) {}

Domain::Domain(std::initializer_list<value_type> il) : Domain() {
    for(auto&& x : il) pimpl_().insert(std::move(x));
}

Domain::Domain(const Domain& rhs) :
  m_pimpl_(std::make_unique<pimpl_type>(rhs.pimpl_())) {}

Domain::Domain(Domain&& rhs) noexcept : m_pimpl_(std::move(rhs.m_pimpl_)) {}

Domain& Domain::operator=(const Domain& rhs) {
    if(this == &rhs) return *this;
    m_pimpl_ = std::make_unique<pimpl_type>(rhs.pimpl_());
    return *this;
}

Domain& Domain::operator=(Domain&& rhs) noexcept {
    if(this == &rhs) return *this;
    m_pimpl_ = std::move(rhs.m_pimpl_);
    return *this;
}

Domain::~Domain() noexcept = default;

//------------------------------------------------------------------------------
//                              Accessors
//------------------------------------------------------------------------------

typename Domain::size_type Domain::rank() const noexcept {
    return m_pimpl_ ? pimpl_().rank() : 0;
}

typename Domain::size_type Domain::size() const noexcept {
    return m_pimpl_ ? pimpl_().size() : 0;
}

std::vector<typename Domain::size_type> Domain::result_extents() const {
    return m_pimpl_ ? pimpl_().result_extents() : std::vector<size_type>{};
}

typename Domain::value_type Domain::result_index(const value_type& old) const {
    if(empty()) throw std::out_of_range("Domain is empty");
    return pimpl_().result_index(old);
}

bool Domain::count(const_reference idx) const noexcept {
    if(!m_pimpl_) return false;
    return pimpl_().count(idx);
}

typename Domain::value_type Domain::operator[](size_type i) const {
    return pimpl_().at(i);
}

//------------------------------------------------------------------------------
//                                  Setters
//------------------------------------------------------------------------------

void Domain::insert(value_type idx) {
    if(!m_pimpl_) m_pimpl_ = std::make_unique<pimpl_type>();
    pimpl_().insert(std::move(idx));
}

Domain Domain::inject(const std::map<size_type, size_type>& injections) const {
    using vector_type = typename Index::index_type;

    if(empty()) return *this;

    // The rank of the indices in the resulting Domain
    const auto out_rank = rank() + injections.size();

    // If we have rank r indices and we are given n injections, we will make a
    // a rank r + n index, hence all modes in the input must be in the range
    // [0, n].
    for(const auto& [k, v] : injections) {
        if(k > out_rank) {
            throw std::out_of_range("Mode " + std::to_string(k) +
                                    "  is not in the range [0, " +
                                    std::to_string(out_rank) + "]. ");
        }
    }

    Domain rv;

    for(const auto& idx : *this) {
        vector_type new_idx(out_rank, 0);
        for(std::size_t i = 0, counter = 0; i < out_rank; ++i) {
            if(injections.count(i))
                new_idx[i] = injections.at(i);
            else {
                new_idx[i] = idx[counter];
                ++counter;
            }
        }
        rv.insert(value_type(std::move(new_idx)));
    }
    return rv;
}

Domain& Domain::operator*=(const Domain& rhs) {
    if(!rhs.m_pimpl_ || !m_pimpl_) {
        m_pimpl_ = std::make_unique<pimpl_type>();
        return *this;
    }

    (*m_pimpl_) *= (*rhs.m_pimpl_);
    return *this;
}

Domain Domain::operator*(const Domain& rhs) const {
    Domain rv(*this);
    rv *= rhs;
    return rv;
}

Domain& Domain::operator+=(const Domain& rhs) {
    if(!m_pimpl_) m_pimpl_ = std::make_unique<pimpl_type>();
    if(!rhs.m_pimpl_) return *this;

    (*m_pimpl_) += (*rhs.m_pimpl_);
    return *this;
}

Domain Domain::operator+(const Domain& rhs) const {
    Domain rv(*this);
    rv += rhs;
    return rv;
}

Domain& Domain::operator^=(const Domain& rhs) {
    if(!m_pimpl_ || !rhs.m_pimpl_) {
        m_pimpl_ = std::make_unique<pimpl_type>();
        return *this;
    }
    (*m_pimpl_) ^= (*rhs.m_pimpl_);
    return *this;
}

Domain Domain::operator^(const Domain& rhs) const {
    Domain rv(*this);
    rv ^= rhs;
    return rv;
}

//------------------------------------------------------------------------------
//                                Utilities
//------------------------------------------------------------------------------

bool Domain::operator==(const Domain& rhs) const noexcept {
    if(!m_pimpl_)
        return !rhs.m_pimpl_;
    else if(!rhs.m_pimpl_)
        return false;
    return *m_pimpl_ == *rhs.m_pimpl_;
}

void Domain::hash(tensorwrapper::detail_::Hasher& h) const { h(pimpl_()); }

std::ostream& Domain::print(std::ostream& os) const {
    os << "{";
    using utilities::printing::operator<<;
    auto begin_itr = begin();
    auto end_itr   = end();
    if(begin_itr != end_itr) {
        os << *begin_itr;
        ++begin_itr;
    }
    while(begin_itr != end_itr) {
        os << ", " << *begin_itr;
        ++begin_itr;
    }
    os << "}";
    return os;
}

//------------------------------------------------------------------------------
//                              Private Methods
//------------------------------------------------------------------------------

typename Domain::pimpl_type& Domain::pimpl_() {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error("PIMPL not set. Did you move from this instance?");
}

const typename Domain::pimpl_type& Domain::pimpl_() const {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error("PIMPL not set. Did you move from this instance?");
}

} // namespace tensorwrapper::sparse_map
