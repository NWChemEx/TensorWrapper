/*
 * Copyright 2024 NWChemEx-Project
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

#include <algorithm>
#include <iterator>
#include <tensorwrapper/symmetry/permutation.hpp>

namespace tensorwrapper::symmetry {

// -----------------------------------------------------------------------------
// -- Ctors, assignment, and dtors
// -----------------------------------------------------------------------------

Permutation::Permutation(cycle_set_initializer_list cycles) :
  Permutation([&]() {
      cycle_container_type rv;
      for(auto& cycle : cycles) rv.emplace(cycle.begin(), cycle.end());
      return rv;
  }()) {}

// -----------------------------------------------------------------------------
// -- Getters
// -----------------------------------------------------------------------------

Permutation::mode_index_type Permutation::minimum_rank() const noexcept {
    if(m_cycles_.empty()) return mode_index_type(0);

    mode_index_type the_max(1);
    for(const auto& cycle : m_cycles_) {
        mode_index_type cycle_max =
          *std::max_element(cycle.begin(), cycle.end());
        mode_index_type cycle_max_plus_one(cycle_max + mode_index_type(1));
        the_max = std::max(cycle_max_plus_one, the_max);
    }
    return the_max;
}

Permutation::cycle_type Permutation::operator[](
  mode_index_type i) const noexcept {
    auto itr = m_cycles_.begin();
    std::advance(itr, i);
    return *itr;
}

void Permutation::valid_offset_(mode_index_type i) const {
    if(i < size()) return;
    auto i_str = std::to_string(i);
    auto n_str = std::to_string(size());
    throw std::out_of_range("input " + i_str + " is not in the range [0," +
                            n_str + ").");
}

void Permutation::is_valid_cycle_(cycle_type cycle) {
    // No element can appear more than once
    std::set<mode_index_type> buffer(cycle.begin(), cycle.end());
    if(buffer.size() == cycle.size()) return;
    throw std::runtime_error("Cycle contains a repeated mode offset");
}

void Permutation::verify_valid_cycle_set_(const cycle_container_type& cycles) {
    for(const auto& lhs : cycles) {
        std::set<mode_index_type> buffer(lhs.begin(), lhs.end());
        for(const auto& rhs : cycles) {
            if(rhs <= lhs) continue;
            for(const auto& x : rhs)
                if(buffer.count(x))
                    throw std::runtime_error("cycles are not disjoint");
        }
    }
}

Permutation::cycle_type Permutation::canonicalize_cycle_(cycle_type cycle) {
    is_valid_cycle_(cycle); // Ensures there's no duplicates

    // 1. Find the smallest mode order
    auto min_itr = std::min_element(cycle.begin(), cycle.end());
    if(min_itr == cycle.begin()) return cycle; // Already canonical

    cycle_type rv;
    auto min_value = *min_itr;

    // Add elements from [max_itr, cycle.end())
    while(min_itr != cycle.end()) {
        rv.push_back(*min_itr);
        ++min_itr;
    }

    // Add elements from [cycle.begin(), max_value)
    auto begin = cycle.begin();
    while(begin != cycle.end()) {
        if(*begin == min_value) break;
        rv.push_back(*begin);
        ++begin;
    }
    return rv;
}

Permutation::cycle_container_type Permutation::remove_trivial_cycles_(
  cycle_container_type input) {
    cycle_container_type rv;

    for(const auto& x : input)
        if(x.size() > 1) rv.insert(canonicalize_cycle_(x));

    verify_valid_cycle_set_(rv);
    return rv;
}

} // namespace tensorwrapper::symmetry
