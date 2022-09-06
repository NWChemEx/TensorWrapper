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
#include "tensorwrapper/detail_/hashing.hpp"
#include "tensorwrapper/sparse_map/index.hpp"
#include <boost/container/flat_set.hpp>
#include <memory>
#include <set>
#include <utilities/iter_tools.hpp>

namespace boost::container {
template<typename ElementType>
void hash_object(const boost::container::flat_set<ElementType>& v,
                 tensorwrapper::detail_::Hasher& h) {
    for(const auto& x : v) h(x);
}
} // namespace boost::container

namespace tensorwrapper::sparse_map::detail_ {

/** @brief Class which holds the state of the Domain.
 *
 *  The Domain holds a set of indices. There's a variety of ways that a Domain
 *  instance could actually hold these indices and we leave it up to the PIMPL
 *  to choose how this is done. The DomainPIMPL class assumes that all indices
 *  are explicitly stored.
 *
 *  @note Once an index has been inserted into a Domain it can only be retrieved
 *        in a read-only fashion (this permits the Domain to be implemented in
 *        a manner which does not explicitly store every index).
 *
 *  @note For developers. Any method which changes the state of the Domain needs
 *        to ensure that it also updates the mode map correctly. See the
 *        `update_mode_map` method for more details.
 */

class DomainPIMPL {
public:
    /// The type used for offsets and indexing
    using size_type = std::size_t;

    /// The type used to store an index
    using value_type = Index;

    /// The type of a read-only reference to an index
    using const_reference = const value_type&;

    /// The type returned by result_extents
    using extents_type = std::vector<size_type>;

    /** @brief Makes an empty Domain
     *
     *  The default ctor makes a Domain which contains no indices. By convention
     *  we say it is of rank 0 even though it does not contain rank 0 indices.
     *
     *  @throw None No throw guarantee.
     */
    DomainPIMPL() = default;

    /// Default copy and move ctors
    DomainPIMPL(const DomainPIMPL& other)     = default;
    DomainPIMPL(DomainPIMPL&& other) noexcept = default;
    DomainPIMPL& operator=(const DomainPIMPL& other) = default;
    DomainPIMPL& operator=(DomainPIMPL&& other) noexcept = default;

    /// Default dtor
    virtual ~DomainPIMPL() noexcept = default;

    /** @brief Determines if the specified index is in the domain or not.
     *
     *  Domains act like sets thus a given index is either in the Domain or not.
     *  The count function can be used to determine if a specific index is in
     *  the Domain or not.
     *
     * @param[in] idx The index whose membership in the domain is in question.
     *
     * @return True if the index is in the domain and false otherwise.
     *
     * @throw None No throw guarantee.
     */
    bool count(const_reference idx) const noexcept;

    /** @brief The rank of the indices in the Domain.
     *
     *  Domains are sets of indices. Each index is made up of @f$N@f$ components
     *  where @f$N@f$ is its rank. All the indices in a Domain must have the
     *  same rank. This function can be used to inquire as to what that rank is.
     *
     *  @note In order to be no-throw this function will return 0 if there are
     *        no indices in the domain. Hence a return of 0 does not necessarily
     *        mean that the domain contains a rank 0 index.
     *
     *  @return The rank of the indices in this domain.
     *
     *  @throw None No throw guarantee.
     */
    size_type rank() const noexcept;

    /** @brief Returns the number of indices in the Domain.
     *
     *  This function is used to determine the number of indices in the Domain.
     *  This is not the size (i.e., rank) of the indices in this Domain.
     *
     *  @return The number of indices in this Domain.
     *
     *  @throw None No throw guarantee.
     */
    size_type size() const noexcept { return m_domain_.size(); }

    /** @brief Returns a vector of extents for the tensor that can be formed
     *         from this Domain.
     *
     *  This function will compute the extents (in terms of tiles or elements
     *  depending on `Index`) for the tensor that results from deleting all
     *  elements besides those in this Domain.
     *
     *  @return A vector such that the `i`-th element is the extent of the
     *          `i`-th mode of the tensor resulting from this Domain.
     *  @throw std::bad_alloc if there is insufficient memory to create the
     *                        result. Strong throw guarantee.
     */
    std::vector<size_type> result_extents() const;

    /** @brief Converts an index in the Domain to its equivalent value in the
     *         inner tensor of the resulting tensor-of-tensors.
     *
     * Each domain instance stores the indices which will be extracted into and
     * made into one of the inner tensors in a tensor-of-tensors. This function
     * will map the original index of an element to its new index in the
     * tensor-of-tensors.
     *
     * @note adding additional elements to domain will in general invalidate
     *       previous calls to this function. It is recommended that you
     *       completely fill the Domain before calling this function.
     *
     * @param[in] old The original index.
     * @return The index in the tensor-of-tensors @p old maps to.
     *
     * @throw std::out_of_range if @p old is not in this Domain. Strong throw
     *                          guarantee.
     * @throw std::bad_alloc if there is insufficient memory to create the index
     *                       for the new tensor. Strong throw guarantee.
     */
    value_type result_index(const value_type& old) const;

    /** @brief Returns the @p i -th index in the Domain
     *
     *  Domains are ordered meaning that any given time it makes sense to speak
     *  of say the zeroth, first, second, etc. index. That said the order can
     *  change upon insertion of a new index as the indices are also sorted.
     *  This function is used to retrieve a read-only reference to the @p i -th
     *  index.
     *
     *  @param[in] i The offset of the requested index. @p i must be in the
     *               range [0, size()).
     *
     *  @return A copy of the @p i-th index.
     *  @throw std::out_of_range if @p i is not in the range [0, size()). Strong
     *                           throw guarantee.
     */
    value_type at(size_type i) const;

    /** @brief Adds @p idx to the Domain
     *
     *  @param[in] idx The index we are adding to the Domain.
     *
     *  Wthrow std::runtime_error If the rank of the index does not match the
     *                            rank of the indices already in the Domain.
     *                            Strong throw guarantee.
     *  @throw std::bad_alloc if there is insufficient memory to store the
     *                        index. Weak throw guarantee.
     */
    void insert(value_type idx);

    /** @brief Makes this instance the Cartesian product of this Domain and
     *         the provided Domain.
     *
     *  The Cartesian product of a Domain, @f$A@f$, containing rank @f$r_A@f$
     *  indices and a Domain, @f$B@f$, containing rank @f$r_B@f$ indices is a
     *  Domain, @f$C@f$ with rank @f$n_C = n_A + n_B@f$ indices, such that:
     *
     *  @f[
     *  C = \left\lbrace(a_i, b_j) \forall a_i\in A
     *                             \forall b_j in B\right\rbrace,
     *  @f]
     *
     *  where @f$(a_i, b_j)@f$ is the tuple made from concatenating the
     *  @f$i@f$-th index in @f$A@f$ with the @f$j@f$-th index in @f$B@f$. The
     *  number of indices in the new domain is equal to the number of indices in
     *  @f$A@f$ times the number of inidces in @f$B@f$. Note that this means
     *  that the Cartesian product of any domain with an empty domain is also
     *  the empty domain.
     *
     * @param[in] rhs The Domain we are taking the Cartesian product with.
     * @return The current Domain with its state set to the Cartesian product of
     *         its previous state and @p rhs.
     *
     * @throw std::bad_alloc if there is insufficient memory to store the new
     *                       state. Strong throw guarantee.
     */
    DomainPIMPL& operator*=(const DomainPIMPL& other);

    /** @brief Sets this Domain to the union of this Domain the provided Domain.
     *
     *  The union of two domains is the Domain containing all indices which
     *  appear in at least one of the two domains. More formally, given two
     *  Domains, @f$A@f$ and @f$B@f$, which both contain indices of rank
     *  @f$r@f$, the union of @f$A@f$ and @f$B@f$ is another Domain @f$C@f$
     *  such that:
     *
     *  @f[
     *  C = \left\lbrace c_i : c_i \in A \lor c_i \in B \right\rbrace
     *  @f]
     *
     *  hence the indices in @f$C@f$ are also of rank @f$r@f$.
     *
     *
     *  @param[in] rhs The Domain we are taking the union with. Must be empty or
     *                 contain indices with the same rank as this Domain.
     *
     *  @return The current Domain with its state overwritten by the union of
     *          its previous state with @p rhs.
     *
     *  @throw std::bad_alloc if there is insufficient memory to store the new
     *                        state. Strong throw guarantee.
     *
     *  @throw std::runtime_error if the indices in @p rhs do not have the same
     *                            rank as the indices in this Domain. Strong
     *                            throw guarantee.
     */
    DomainPIMPL& operator+=(const DomainPIMPL& other);

    /** @brief Makes this Domain the intersection of this Domain and @p rhs.
     *
     *  Given a Domain, @f$A@f$, and a Domain @f$B@f$, the intersection of
     *  @f$A@f$ with @f$B@f$ is another Domain @f$C@f$ given by:
     *
     *  @f[
     *  C = \left\lbrace c_i | c_i \in A \land c_i \in B \right\rbrace
     *  @f]
     *
     *  Of note, the intersection of two Domains with indices of different rank
     *  is the empty domain.
     *
     *  @param[in] rhs The Domain we are taking the intersection with.
     *
     *  @return The current Domain set to the intersection of its previous state
     *          with @p rhs.
     *
     *  @throw std::bad_alloc if there is insufficient memory to allocate the
     *                        new state. Strong throw guarantee.
     */
    DomainPIMPL& operator^=(const DomainPIMPL& other);

    /** @brief Compares two DomainPIMPL instances for exact equality.
     *
     *  Two DomainPIMPL instances are equal if they contain the same set of
     *  indices, regardless of whether the individual indices are actually
     *  stored in memory.
     *
     *  @param[in] rhs The DomainPIMPL on the right side of the operator.
     *
     *  @return True if the two instances contain the same indices and false
     *          otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const DomainPIMPL& rhs) const noexcept;

    /** @brief Computes the hash of the Domain.
     *
     *  @param[in,out] h The object computing the hash. After this call the
     *                   internal state will be updated with the hash of the
     *                   Domain's state.
     */
    void hash(tensorwrapper::detail_::Hasher& h) const { h(m_domain_); }

private:
    /** @brief Adds the specified index to the mode_map instance.
     *
     *  The internal mode map member is used to map from the indices in this
     *  Domain to the indices in the resulting tensor-of-tensors. It is
     *  essential that the mode map be updated anytime the contents of the
     *  Domain change.
     *
     * @param[in] idx The index we are adding to mode map.
     *
     * @throw std::bad_alloc if there is insufficient memory to add the index.
     *                       Weak throw guarantee.
     */
    void update_mode_map(const_reference idx);

    /// Ensures that @p i is in the range [0, size())
    void bounds_check_(size_type i) const;

    /// The instance actually holding the state.
    boost::container::flat_set<value_type> m_domain_;

    /// Keeps track of the offsets per mode of the indices in this Domain
    std::vector<std::set<size_type>> m_mode_map_;
}; // class DomainPIMPL

//------------------------------------------------------------------------------
//                      Related Free Functions
//------------------------------------------------------------------------------

/** @brief Adds a string representation of this DomainPIMPL to the provided
 *         stream.
 *  @relates Domain
 *
 *  @param[in,out] os The stream we are adding this DomainPIMPL to. After this
 *                    call @p os will contain a string representation of
 *                    this DomainPIMPL.
 *  @param[in] d The pimpl to print.
 *  @return @p os after adding this Domain to it.
 */
inline std::ostream& operator<<(std::ostream& os, const DomainPIMPL& p) {
    os << "{";
    for(std::size_t i = 0; i < p.size(); ++i) {
        os << p.at(i);
        if(i + 1 != p.size()) os << ", ";
    }
    return os << "}";
}

/** @brief Determines if two DomainPIMPL instances are different
 *
 *  Two DomainPIMPL instances are equal if they contain the same set of
 *  indices, regardless of whether the individual indices are actually
 *  stored in memory.
 *
 *  @tparam Index The type of index in the domain.
 *
 *  @param[in] lhs The DomainPIMPL on the left side of the operator
 *  @param[in] rhs The DomainPIMPL on the right side of the operator
 *
 *  @return True if the two instances are different and false otherwise.
 *
 *  @throw None No throw guarantee.
 */

inline bool operator!=(const DomainPIMPL& lhs, const DomainPIMPL& rhs) {
    return !(lhs == rhs);
}

//------------------------------------------------------------------------------
// Inline implementations
//------------------------------------------------------------------------------

inline bool DomainPIMPL::count(const_reference idx) const noexcept {
    return m_domain_.count(idx);
}

inline typename DomainPIMPL::size_type DomainPIMPL::rank() const noexcept {
    return m_domain_.empty() ? 0 : m_domain_.begin()->size();
}

inline std::vector<typename DomainPIMPL::size_type>
DomainPIMPL::result_extents() const {
    std::vector<size_type> rv(rank(), 0);
    for(size_type i = 0; i < rank(); ++i) rv[i] = m_mode_map_[i].size();
    return rv;
}

inline typename DomainPIMPL::value_type DomainPIMPL::result_index(
  const value_type& old) const {
    if(size() == 0 || old.size() != rank())
        throw std::out_of_range("Index is not in domain");
    std::vector<size_type> rv(rank(), 0);
    for(std::size_t i = 0; i < rank(); ++i) {
        const auto& offsets = m_mode_map_[i];
        const auto value    = old[i];
        auto itr            = std::find(offsets.begin(), offsets.end(), value);
        if(itr == offsets.end())
            throw std::out_of_range("Index is not in domain");
        rv[i] = std::distance(offsets.begin(), itr);
    }
    return Index(rv.begin(), rv.end());
}

inline typename DomainPIMPL::value_type DomainPIMPL::at(size_type i) const {
    bounds_check_(i);
    return *(m_domain_.begin() + i);
}

inline void DomainPIMPL::update_mode_map(const_reference idx) {
    if(m_mode_map_.empty()) {
        std::vector<std::set<size_type>>(idx.size()).swap(m_mode_map_);
    }
    for(std::size_t i = 0; i < idx.size(); ++i) m_mode_map_[i].insert(idx[i]);
}

inline void DomainPIMPL::insert(value_type idx) {
    if(!m_domain_.empty() && idx.size() != rank()) {
        using namespace std::string_literals;
        throw std::runtime_error("Rank of idx ("s + std::to_string(idx.size()) +
                                 ") != rank of domain ("s +
                                 std::to_string(rank()) + ")"s);
    }
    if(count(idx)) return;
    update_mode_map(idx);
    m_domain_.insert(idx);
}

inline DomainPIMPL& DomainPIMPL::operator*=(const DomainPIMPL& other) {
    const bool is_empty = m_domain_.empty() || other.m_domain_.empty();

    if(is_empty) {
        m_domain_.clear();
        m_mode_map_.clear();
        return *this;
    }

    boost::container::flat_set<value_type> buffer;
    std::vector<std::set<size_type>> new_modes(m_mode_map_);
    const auto& other_mm = other.m_mode_map_;
    new_modes.insert(new_modes.end(), other_mm.begin(), other_mm.end());

    std::vector<size_type> new_idx(rank() + other.rank());
    for(const auto& x : m_domain_) {
        for(const auto& [i, xi] : utilities::Enumerate(x)) new_idx[i] = xi;
        for(const auto& y : other.m_domain_) {
            for(const auto& [i, yi] : utilities::Enumerate(y))
                new_idx[i + rank()] = yi;
            buffer.insert(value_type(new_idx));
        }
    }
    m_domain_.swap(buffer);
    m_mode_map_.swap(new_modes);
    return *this;
}

inline DomainPIMPL& DomainPIMPL::operator+=(const DomainPIMPL& other) {
    if(other.m_domain_.empty())
        return *this;
    else if(m_domain_.empty())
        return (*this) = other;

    // They're not empty so they need to have the same rank
    if(rank() != other.rank())
        throw std::runtime_error("Intersection requires ranks to be the same");

    for(const auto& x : other.m_domain_) insert(x);
    return *this;
}

inline DomainPIMPL& DomainPIMPL::operator^=(const DomainPIMPL& other) {
    if(this == &other) return *this;

    const bool is_empty   = m_domain_.empty() || other.m_domain_.empty();
    const bool diff_ranks = rank() != other.rank();

    if(is_empty || diff_ranks) {
        m_domain_.clear();
        m_mode_map_.clear();
        return *this;
    }

    boost::container::flat_set<value_type> old_domain(m_domain_);
    m_domain_.clear();
    m_mode_map_.clear();
    for(const auto& x : old_domain)
        if(other.count(x)) insert(x);

    return *this;
}

inline bool DomainPIMPL::operator==(const DomainPIMPL& rhs) const noexcept {
    if(rank() != rhs.rank()) return false;
    if(size() != rhs.size()) return false;
    return m_domain_ == rhs.m_domain_;
}

inline void DomainPIMPL::bounds_check_(size_type i) const {
    if(i < size()) return;
    using namespace std::string_literals;
    throw std::out_of_range("i = "s + std::to_string(i) +
                            " is not in the range [0, "s +
                            std::to_string(size()) + ")."s);
}

} // namespace tensorwrapper::sparse_map::detail_
