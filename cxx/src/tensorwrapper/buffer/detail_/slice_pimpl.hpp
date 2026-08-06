/*
 * Copyright 2026 NWChemEx-Project
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
#include "replicated_view_pimpl.hpp"
#include <stdexcept>

namespace tensorwrapper::buffer::detail_ {
/** @brief PIMPL holding a non-owning pointer to a Replicated object and slice
 *         bounds.
 *
 *  Implements a view of a slice of a Replicated buffer. Slice indices are
 *  relative to the view; index translation to the underlying Replicated is
 *  performed in get_elem and set_elem.
 */
template<typename ReplicatedType>
class SlicePIMPL : public ReplicatedViewPIMPL<ReplicatedType> {
private:
    using my_base = ReplicatedViewPIMPL<ReplicatedType>;

public:
    /// Pull in types from base
    ///@{
    using typename my_base::const_element_reference;
    using typename my_base::const_layout_reference;
    using typename my_base::const_slice_type;
    using typename my_base::element_type;
    using typename my_base::index_vector;
    using typename my_base::layout_reference;
    using typename my_base::layout_type;
    using typename my_base::pimpl_pointer;
    using typename my_base::size_type;
    using typename my_base::slice_type;
    ///@}

    /// Pull in base's ctors
    using my_base::my_base;

    /** @brief Creates a PIMPL that views a slice of @p replicated_ptr.
     *
     *  @param[in] replicated_ptr Non-owning pointer to the Replicated object
     *                            (or nullptr for an empty view).
     *  @param[in] first_elem     Indices of the first element in the slice
     *                            (inclusive).
     *  @param[in] last_elem      Indices of the first element not in the slice
     *                            (exclusive).
     */
    SlicePIMPL(ReplicatedType* replicated_ptr, index_vector first_elem,
               index_vector last_elem) :
      m_replicated_ptr_(replicated_ptr),
      m_first_elem_(std::move(first_elem)),
      m_last_elem_(std::move(last_elem)),
      m_layout_ptr_(replicated_ptr ?
                      replicated_ptr->layout()
                        .slice(m_first_elem_.begin(), m_first_elem_.end(),
                               m_last_elem_.begin(), m_last_elem_.end())
                        .template clone_as<layout_type>() :
                      nullptr) {}

protected:
    pimpl_pointer clone_() const override {
        return std::make_unique<SlicePIMPL>(m_replicated_ptr_, m_first_elem_,
                                            m_last_elem_);
    }

    layout_reference layout_() override { return *m_layout_ptr_; }

    const_layout_reference layout_() const override { return *m_layout_ptr_; }

    const_element_reference get_elem_(
      const index_vector& slice_index) const override {
        return replicated().get_elem(translate_index_(slice_index));
    }

    void set_elem_(const index_vector& slice_index,
                   element_type value) override {
        if constexpr(std::is_const_v<ReplicatedType>) {
            throw std::runtime_error(
              "Cannot set elements of a const ReplicatedViewPIMPL.");
        } else {
            auto new_index = translate_index_(slice_index);
            replicated().set_elem(new_index, std::move(value));
        }
    }

    slice_type slice_(const index_vector& first_elem,
                      const index_vector& last_elem) override {
        auto new_first_elem = translate_index_(first_elem);
        auto new_last_elem  = translate_index_(last_elem);
        return slice_type(*m_replicated_ptr_, new_first_elem, new_last_elem);
    }

    const_slice_type slice_(const index_vector& first_elem,
                            const index_vector& last_elem) const override {
        auto new_first_elem = translate_index_(first_elem);
        auto new_last_elem  = translate_index_(last_elem);
        return const_slice_type(*m_replicated_ptr_, new_first_elem,
                                new_last_elem);
    }

private:
    ReplicatedType* m_replicated_ptr_;
    index_vector m_first_elem_;
    index_vector m_last_elem_;

    std::unique_ptr<layout_type> m_layout_ptr_;

    void assert_replicated_ptr_() const {
        if(m_replicated_ptr_ == nullptr) {
            throw std::runtime_error(
              "SlicePIMPL has no Replicated object. Was it default "
              "initialized?");
        }
    }

    ReplicatedType& replicated() {
        assert_replicated_ptr_();
        return *m_replicated_ptr_;
    }

    const ReplicatedType& replicated() const {
        assert_replicated_ptr_();
        return *m_replicated_ptr_;
    }

    index_vector translate_index_(const index_vector& slice_index) const {
        index_vector result;
        result.reserve(slice_index.size());
        for(size_type i = 0; i < slice_index.size(); ++i) {
            result.push_back(m_first_elem_[i] + slice_index[i]);
        }
        return result;
    }
};

} // namespace tensorwrapper::buffer::detail_
