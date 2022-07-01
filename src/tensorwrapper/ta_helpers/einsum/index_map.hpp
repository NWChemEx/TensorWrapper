#pragma once
#include "index_utils.hpp"
#include "types.hpp"

namespace tensorwrapper::ta_helpers::einsum {

class IndexMap {
public:
    IndexMap() noexcept = default;
    IndexMap(types::index result_idx, types::index lhs_idx,
             types::index rhs_idx);
    ~IndexMap() noexcept = default;

    template<typename T>
    auto select_result(T&& quantities) const;

    template<typename T>
    auto select_lhs(T&& quantities) const;

    template<typename T>
    auto select_rhs(T&& quantities) const;

    template<typename T>
    auto select(T&& quantities) const;

    const auto& result_vars() const { return m_result_vars_; }
    const auto& lhs_vars() const { return m_lhs_vars_; }
    const auto& rhs_vars() const { return m_rhs_vars_; }

    bool operator==(const IndexMap& other) const noexcept;
    bool operator!=(const IndexMap& other) const noexcept;

private:
    template<typename T>
    auto select_(const types::index_set& indices, T&& quantities) const;

    types::index_set m_result_vars_;
    types::index_set m_lhs_vars_;
    types::index_set m_rhs_vars_;
}; // class IndexMap

inline IndexMap::IndexMap(types::index result_idx, types::index lhs_idx,
                          types::index rhs_idx) :
  m_result_vars_(parse_index(std::move(result_idx))),
  m_lhs_vars_(parse_index(std::move(lhs_idx))),
  m_rhs_vars_(parse_index(std::move(rhs_idx))) {}

template<typename T>
inline auto IndexMap::select_result(T&& quantities) const {
    return select_(result_vars(), std::forward<T>(quantities));
}

template<typename T>
inline auto IndexMap::select_lhs(T&& quantities) const {
    return select_(lhs_vars(), std::forward<T>(quantities));
}

template<typename T>
inline auto IndexMap::select_rhs(T&& quantities) const {
    return select_(rhs_vars(), std::forward<T>(quantities));
}

template<typename T>
inline auto IndexMap::select(T&& qs) const {
    return std::make_tuple(select_result(qs), select_lhs(qs), select_rhs(qs));
}

inline bool IndexMap::operator==(const IndexMap& other) const noexcept {
    return std::tie(m_result_vars_, m_lhs_vars_, m_rhs_vars_) ==
           std::tie(other.m_result_vars_, other.m_lhs_vars_, other.m_rhs_vars_);
}

inline bool IndexMap::operator!=(const IndexMap& other) const noexcept {
    return !((*this) == other);
}

template<typename T>
inline auto IndexMap::select_(const types::index_set& indices,
                              T&& quantities) const {
    using result_t = decltype(quantities.at(indices[0]));
    using clean_t  = std::decay_t<result_t>;
    std::vector<clean_t> rv;
    for(auto x : indices) rv.push_back(quantities.at(x));
    return rv;
}

} // namespace tensorwrapper::ta_helpers::einsum