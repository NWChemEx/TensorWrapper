/*
 * Copyright 2025 NWChemEx-Project
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
#include <tensorwrapper/dsl/dummy_indices.hpp>

namespace tensorwrapper::buffer {

/** @brief Class for working out details pertaining to a tensor contraction.
 *
 *  N.B. Contraction covers direct product (which is a special case of
 *  contraction with 0 dummy indices).
 */
class ContractionPlanner {
public:
    /// String type users use to label modes
    using string_type = std::string;

    /// Type of the parsed labels
    using label_type = dsl::DummyIndices<string_type>;

    ContractionPlanner(string_type result, string_type lhs, string_type rhs) :
      ContractionPlanner(label_type(result), label_type(lhs), label_type(rhs)) {
    }

    ContractionPlanner(label_type result, label_type lhs, label_type rhs) :
      m_result_(std::move(result)),
      m_lhs_(std::move(lhs)),
      m_rhs_(std::move(rhs)) {
        assert_no_repeated_indices_();
        assert_dummy_indices_are_similar_();
        assert_no_shared_free_();
    }

    /// Labels in LHS that are NOT summed over
    label_type lhs_free() const { return m_lhs_.intersection(m_result_); }

    /// Labels in RHS that are NOT summed over
    label_type rhs_free() const { return m_rhs_.intersection(m_result_); }

    /// Labels in LHS that ARE summed over
    label_type lhs_dummy() const { return m_lhs_.difference(m_result_); }

    /// Labels in RHS that ARE summed over
    label_type rhs_dummy() const { return m_rhs_.difference(m_result_); }

    /** @brief LHS permuted so free indices are followed by dummy indices. */
    label_type lhs_permutation() const {
        using split_string_type = typename label_type::split_string_type;
        split_string_type rv;
        auto lfree  = lhs_free();
        auto ldummy = lhs_dummy();
        for(const auto& freei : m_result_) {
            if(!lfree.count(freei)) continue;
            rv.push_back(freei);
        }
        for(const auto& dummyi : ldummy) rv.push_back(dummyi);
        return label_type(std::move(rv));
    }

    /** @brief RHS permuted so dummy indices are followed by free indices. */
    label_type rhs_permutation() const {
        typename label_type::split_string_type rv;
        auto rfree  = rhs_free();
        auto rdummy = lhs_dummy(); // Use LHS dummy to get the same order!
        for(const auto& dummyi : rdummy)
            rv.push_back(dummyi); // Know it only appears 1x
        for(const auto& freei : m_result_) {
            if(!rfree.count(freei)) continue;
            rv.push_back(freei); // Know it only appears 1x
        }
        return label_type(std::move(rv));
    }

private:
    /// Ensures no tensor contains a repeated label
    void assert_no_repeated_indices_() const {
        const bool result_good = !m_result_.has_repeated_indices();
        const bool lhs_good    = !m_lhs_.has_repeated_indices();
        const bool rhs_good    = !m_rhs_.has_repeated_indices();

        if(result_good && lhs_good && rhs_good) return;
        throw std::runtime_error("One or more terms contain repeated labels");
    }

    /// Ensures the dummy indices are permutations of each other
    void assert_dummy_indices_are_similar_() const {
        if(lhs_dummy().is_permutation(rhs_dummy())) return;
        throw std::runtime_error("Dummy indices must appear in all terms");
    }

    /// Asserts LHS and RHS do not share free indices, which is Hadamard-product
    void assert_no_shared_free_() const {
        if(!lhs_free().intersection(rhs_free()).size()) return;
        throw std::runtime_error("Contraction must sum repeated indices");
    }

    label_type m_result_;
    label_type m_lhs_;
    label_type m_rhs_;
};

} // namespace tensorwrapper::buffer