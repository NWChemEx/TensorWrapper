#pragma once
#include "tensorwrapper/sparse_map/domain/domain.hpp"
#include "tensorwrapper/sparse_map/index.hpp"
#include <memory> // unique_ptr
#include <utilities/printing/print_stl.hpp>

namespace tensorwrapper::sparse_map {

// Forward declaration
class SparseMap;

namespace detail_ {

/** @brief Defines the API for SparseMapPIMPL instances.
 *
 *  The SparseMapPIMPL is in charge of holding the actual state of the SparseMap
 *  and performing basic manipulations on it.
 */
class SparseMapPIMPL {
public:
    /// Type used for counting and offsets
    using size_type = std::size_t;

    /// Type of the independent indices stored in this SparseMap
    using key_type = Index;

    /// Type of the dependent index containing Domains
    using mapped_type = Domain;

    /** @brief Constructs a new empty SparseMap.
     *
     *  This constructor is used to create a new SparseMap which contains no
     *  independent elements (and thus no dependent elements either).
     *
     *  @throw None No throw guarantee
     */
    SparseMapPIMPL() = default;

    /// Default ctors
    SparseMapPIMPL(const SparseMapPIMPL& rhs) = default;
    SparseMapPIMPL(SparseMapPIMPL&& rhs)      = default;
    SparseMapPIMPL& operator=(const SparseMapPIMPL& rhs) = default;
    SparseMapPIMPL& operator=(SparseMapPIMPL&& rhs) = default;

    /// Default polymorphic dtor
    virtual ~SparseMapPIMPL() = default;

    /** @brief Returns the number of independent indices in this SparseMap.
     *
     *  Each independent index in this SparseMap is paired with a Domain. This
     *  function can be used to retrieve the number of independent indices in
     *  this SparseMap, which is also the number of Domains and the number of
     *  independent-index-domain pairs.
     *
     *  @return The number of independent indices in this SparseMap.
     *
     *  @throw None No throw guarantee.
     */
    size_type size() const noexcept { return m_sm_.size(); }

    bool count(const key_type& ind) const noexcept { return m_sm_.count(ind); }

    /** @brief Returns the rank of the independent indices.
     *
     *  All independent indices in the SparseMap must have the same rank
     *  (*i.e.*, number of modes). This function can be used to determine
     *  what the rank of the independent indices are. Note that if the SparseMap
     *  is empty we return a rank of 0; hence both empty sparse maps and
     *  SparseMaps with independent indices of rank 0 return 0 (use `empty()` to
     *  distinguish between the two cases).
     *
     *  @return The rank of the independent indices.
     *
     *  @throw None No throw guarantee.
     */
    size_type ind_rank() const noexcept;

    /** @brief The number of independent modes associated with each Domain.
     *
     *  Each independent index maps to a Domain. Each Domain contains indices of
     *  the same rank. For a given SparseMap all of the mapped to Domains must
     *  contain indices of the same rank (or be empty). This function will
     *  return the rank of the indices in the Domains. The return is 0 if this
     *  SparseMap is empty, all the Domains are empty, or all Domains contain a
     *  rank 0 index.
     *
     *  @return The rank of the indices in the Domains this SparseMap maps to.
     *
     *  @throw None No throw guarantee.
     */
    size_type dep_rank() const noexcept;

    /** @brief Adds a dependent index to an independent index's Domain.
     *
     *  This function adds an dependent index to and independent index's Domain.
     *  This function will create the Domain if the independent index is not
     *  already in the SparseMap. If the dependent index is already in the
     *  independent index's Domain this is a no-op.
     *
     *  @param[in] key The independent index to which @p value should be added.
     *                 If this SparseMap is non-empty the rank of @p key must be
     *                 equal to `ind_rank()`.
     *  @param[in] value The dependent index being added to @p key's domain. If
     *                 @p key's Domain is non-empty the rank of @p value must be
     *                 equal to `dep_rank()`
     *
     *  @throw std::bad_alloc if there is insufficient memory to add the new
     *                        state. Strong throw guarantee.
     *  @throw std::runtime_error if this SparseMap is non-empty and the rank of
     *                            @p key is not equal to `ind_rank()` or if the
     *                            Domain associated with @p key is non-empty and
     *                            the rank of @p value is not equal to
     *                            `dep_rank()`. Strong throw guarantee.
     */
    void add_to_domain(const key_type& ind, const Index& dep);

    /** @brief Returns the @p i-th std::pair<Index, Domain> in the
     *         SparseMap.
     *
     *  The independent-index-domain pairs are stored in an ordered manner. This
     *  function allows one to retrieve the pair they want by offset. It should
     *  be noted that the input to this function is **NOT** used as a key.
     *
     *  @param[in] i Which independent-domain pair to return. Must be in the
     *               range [0, size()).
     *  @return The @p i-th independent-index-domain pair in a read/write state.
     *
     *  @throw std::out_of_range if @p i is not in the range [0, size()). Strong
     *                           throw guarantee.
     */
    auto& at(size_type i);

    /** @brief Returns the @p i-th std::pair<Index, Domain> in the
     *         SparseMap.
     *
     *  The independent-index-domain pairs are stored in an ordered manner. This
     *  function allows one to retrieve the pair they want by offset. It should
     *  be noted that the input to this function is **NOT** used as a key.
     *
     *  @param[in] i Which independent-domain pair to return. Must be in the
     *               range [0, size()).
     *  @return The @p i-th independent-index-domain pair in a read-only state.
     *
     *  @throw std::out_of_range if @p i is not in the range [0, size()). Strong
     *                           throw guarantee.
     */
    const auto& at(size_type i) const;

    /** @brief Returns the Domain associated with the specified independent
     *         index.
     *
     *  This function can be used to retrieve the Domain associated with a
     *  independent index.
     *
     * @param[in] key The independent index whose Domain we want. The rank of
     *                @p key must be equal to `ind_rank()`.
     *
     * @return The Domain associated with @p key in a read-only manner.
     *
     * @throw std::out_of_range if @p key is not in the SparseMap. Strong throw
     *                          guarantee.
     * @throw std::runtime_error if the rank of @p key is not equal to
     *                           `ind_rank()`. Strong throw guarantee.
     */
    const auto& at(const key_type& ind) const;

    /** @brief Sets this SparseMap to the direct product of this SparseMap and
     *         another SparseMap.
     *
     *  Given a SparseMap @f$A@f$ with @f$i@f$-th element @f$(a_i, \alpha_i)@f$
     *  (@f$a_i@f$ is the independent index and @f$\alpha_i@f$ is the Domain
     *  associated with @f$a_i@f$) and a SparseMap @f$B@f$ with @f$j@f$-th
     *  element @f$(b_j, \beta_j)@f$ this function computes a new SparseMap
     *  @f$C@f$ which we say is the direct product of @f$A@f$ with  @f$B@f$.
     *  @f$C@f$ is given by:
     *
     *  @f[
     *  C = \left\lbrace (a_ib_j, \alpha_i\beta_j) \forall (a_i, \alpha_i) \in A
     *                                             \forall (b_j, \beta_j) \in B
     *      \right\rbrace
     *  @f]
     *
     * @param[in] rhs The SparseMap we are taking the direct product with.
     *
     * @return The current SparseMap set to the direct product of this
     *         SparseMap's initial state with @p rhs.
     *
     * @throw std::bad_alloc if there is not enough memory to store the new
     *                       state. Strong throw guarantee.
     */
    auto& direct_product_assign(const SparseMapPIMPL& rhs);

    /** @brief Sets this SparseMap to the SparseMap with domains given by the
     *         Cartesian product of the Domains previously in this SparseMap
     *         with the Domains in @p rhs.
     *
     *  Given a SparseMap @f$A@f$ which maps independent index @f$i@f$ to
     *  @f$a_i@f$ and a SparseMap @f$B@f$ which maps independent index @f$i@f$
     *  to @f$b_i@f$. This function computes a third SparseMap, @f$C@f$, where
     *  the Domain @f$c_i@f$-th element is the Cartesian product of @f$a_i@f$
     *  with @f$b_i@f$.
     *
     * @param[in] rhs The SparseMap we are taking the Cartesian product with.
     *
     * @return The current SparseMap with Domains set to the resultof the
     *         Cartesian product of this SparseMap's previous domains with
     *         @p rhs's Domains.
     *
     * @throw std::bad_alloc if there is not enough memory to store the new
     *                       state. Strong throw guarantee.
     */
    SparseMapPIMPL& operator*=(const SparseMapPIMPL& rhs);

    /** @brief Sets this to the union of this and another map.
     *
     *  Given two SparseMaps sm1(f -> g) and sm2(f -> g), the union is formed by
     *  mapping f_i to any element in g which f_i is mapped to by either sm1 or
     *  sm2.
     *
     *  Requires that either one of the sets is empty or both sets have the same
     *  rank in independent and dependent indices.
     *
     * @param[in] rhs The SparseMap to take the union with this instance.
     *
     * @return The current instance set to the union of the two maps.
     *
     * @throw std::runtime_error if neither map is empty and the rank of the
     *                           independent/dependent indices of this instance
     *                           are not equal to the rank of the independent/
     *                           dependent indices of @p rhs.
     */
    SparseMapPIMPL& operator+=(const SparseMapPIMPL& rhs);

    /** @brief Makes this the intersection of this SparseMap and another map.
     *
     *  Given two SparseMaps sm1(f -> g) and sm2(f -> g), the intersection is
     *  formed by mapping f_i to any element in g which f_i is mapped to by both
     *  sm1 and sm2. Note that if the ranks of the independent indices (or the
     *  dependent indices) are different between the two maps the intersection
     *  is empty. Similarly the intersection of any map with an empty map is
     *  also empty.
     *
     * @param[in] sm The SparseMap to take the intersection with this instance.
     *
     * @return The current SparseMap set to the intersection of the two maps.
     *
     * @throw std::bad_alloc if there is insufficient memory to store the new
     *                       state. Strong throw guarantee.
     */
    SparseMapPIMPL& operator^=(const SparseMapPIMPL& rhs);

    /** @brief Determines if two SparseMaps are identical.
     *
     *  Two SparseMaps are the same if they:
     *  - map from the same type of independent/dependent index
     *    - *e.g.* independent indices ar both ElementalIndex and dependent
     *      indices are both TileIndex
     *  - contain the same number of independent-indices
     *  - the set of independent indices is the same, and
     *  - each independent index maps to the same Domain
     *
     *  @param[in] rhs The other SparseMap to compare to.
     *
     *  @return True if this SparseMap is the same as @p rhs and false
     *          otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const SparseMapPIMPL& rhs) const noexcept;

    /** @brief Adds a string representation of the SparseMap to the stream.
     *
     * @param[in,out] os The stream we are adding the string representation to.
     *                   After this call the stream will contain the string
     *                   representation of the SparseMap.
     *
     * @return @p os with this SparseMap added to it.
     */
    std::ostream& print(std::ostream& os) const;

    /** @brief Adds this SparseMap's state to a hash.
     *
     *  @param[in,out] h The object hashing the SparseMap. After this call the
     *                   internal hash of @p h will be updated to include this
     *                   SparseMap's state.
     */
    void hash(tensorwrapper::detail_::Hasher& h) const { h(m_sm_); }

private:
    /// Type of the std::map holding the SparseMap's state
    using map_type = std::map<key_type, mapped_type>;

    /// The actual data in the SparseMap
    map_type m_sm_;
}; // class SparseMapPIMPL

/** @brief Adds a string representation of the SparseMap to the stream.
 *  @related SparseMapPIMPL
 *
 *  This is a convenience function for calling SparseMapBase::print on a stream.
 *
 *  @param[in,out] os The stream we are adding the string representation to.
 *                   After this call the stream will contain the string
 *                   representation of the SparseMap.
 *  @param[in] smb The SparseMap we are printing.
 *
 *  @return @p os with this SparseMap added to it.
 */
std::ostream& operator<<(std::ostream& os, const SparseMapPIMPL& smb) {
    return smb.print(os);
}

/** @brief Determines if two SparseMaps are different.
 *  @relates SparseMapPIMPL
 *
 *  Two SparseMaps are the same if they:
 *  - contain the same number of independent-indices
 *  - the set of independent indices is the same, and
 *  - each independent index maps to the same Domain
 *
 *  @param[in] lhs The SparseMap on the right side of the operator
 *  @param[in] rhs The SparseMap on the left side of the operator
 *
 *  @return False if this SparseMap is the same as @p rhs and true otherwise.
 *
 *  @throw None No throw guarantee.
 */
bool operator!=(const SparseMapPIMPL& lhs, const SparseMapPIMPL& rhs) {
    return !(lhs == rhs);
}

//------------------------------------------------------------------------------
//                           Inline Implementations
//------------------------------------------------------------------------------
typename SparseMapPIMPL::size_type SparseMapPIMPL::ind_rank() const noexcept {
    return !m_sm_.empty() ? m_sm_.begin()->first.size() : 0;
}

typename SparseMapPIMPL::size_type SparseMapPIMPL::dep_rank() const noexcept {
    for(const auto& [k, v] : m_sm_)
        if(v.rank() > 0) return v.rank();
    return 0; // We get here if it's empty or if all Domains have rank 0
}

void SparseMapPIMPL::add_to_domain(const key_type& ind, const Index& dep) {
    if(!m_sm_.empty() && ind_rank() != ind.size())
        throw std::runtime_error("Independent index");
    else if(!m_sm_.empty() && dep_rank() != dep.size())
        throw std::runtime_error("Dependent index");

    m_sm_[ind].insert(dep);
}

auto& SparseMapPIMPL::at(size_type i) {
    if(i >= size())
        throw std::out_of_range("Offset must be in range [0, size())");
    auto itr = m_sm_.begin();
    std::advance(itr, i);
    return *itr;
}

const auto& SparseMapPIMPL::at(size_type i) const {
    if(i >= size())
        throw std::out_of_range("Offset must be in range [0, size())");
    auto itr = m_sm_.begin();
    std::advance(itr, i);
    return *itr;
}

const auto& SparseMapPIMPL::at(const key_type& ind) const {
    if(ind.size() != ind_rank())
        throw std::runtime_error("Rank of key does not equal ind_rank()");
    return m_sm_.at(ind);
}

std::ostream& SparseMapPIMPL::print(std::ostream& os) const {
    using utilities::printing::operator<<;
    os << m_sm_;
    return os;
}

auto& SparseMapPIMPL::direct_product_assign(const SparseMapPIMPL& rhs) {
    if(m_sm_.empty() || rhs.m_sm_.empty()) {
        m_sm_.clear();
        return *this;
    }
    map_type new_map;

    using vector_type = std::vector<size_type>;
    auto new_rank     = ind_rank() + rhs.ind_rank();
    for(auto [lkey, lval] : m_sm_) {
        for(const auto& [rkey, rval] : rhs.m_sm_) {
            vector_type new_index;
            new_index.reserve(new_rank);
            new_index.insert(new_index.end(), lkey.begin(), lkey.end());
            new_index.insert(new_index.end(), rkey.begin(), rkey.end());
            auto new_domain = lval * rval;
            if(new_domain.empty()) continue;
            key_type key(new_index);
            new_map.emplace(std::move(key), std::move(new_domain));
        }
    }
    m_sm_.swap(new_map);
    return *this;
}

SparseMapPIMPL& SparseMapPIMPL::operator*=(const SparseMapPIMPL& rhs) {
    if(m_sm_.empty())
        return *this;
    else if(rhs.m_sm_.empty()) {
        m_sm_.clear();
        return *this;
    }
    if(ind_rank() != rhs.ind_rank())
        throw std::runtime_error("Independent ranks do not match");

    map_type new_map;
    for(const auto& [lind, ldom] : m_sm_) {
        if(rhs.count(lind)) {
            auto new_dom = ldom * rhs.at(lind);
            if(new_dom.empty()) continue;
            new_map.emplace(lind, std::move(new_dom));
        }
    }
    m_sm_.swap(new_map);
    return *this;
}

SparseMapPIMPL& SparseMapPIMPL::operator+=(const SparseMapPIMPL& rhs) {
    if(rhs.m_sm_.empty())
        return *this;
    else if(m_sm_.empty()) {
        m_sm_ = rhs.m_sm_;
        return *this;
    }

    if(ind_rank() != rhs.ind_rank())
        throw std::runtime_error("Independent index ranks do not match");

    for(const auto& [k, v] : rhs.m_sm_) {
        for(const auto& dep : v) m_sm_[k].insert(dep);
    }
    return *this;
}

SparseMapPIMPL& SparseMapPIMPL::operator^=(const SparseMapPIMPL& rhs) {
    if(m_sm_.empty())
        return *this;
    else if(rhs.m_sm_.empty() || (ind_rank() != rhs.ind_rank())) {
        m_sm_.clear();
        return *this;
    }

    map_type new_map;
    for(const auto& [lind, ldom] : m_sm_) {
        if(!rhs.count(lind)) continue;
        const auto intersection = ldom ^ rhs.at(lind);
        for(const auto& dep : intersection) new_map[lind].insert(dep);
    }
    m_sm_.swap(new_map);
    return *this;
}

bool SparseMapPIMPL::operator==(const SparseMapPIMPL& rhs) const noexcept {
    return m_sm_ == rhs.m_sm_;
}

} // namespace detail_
} // namespace tensorwrapper::sparse_map
