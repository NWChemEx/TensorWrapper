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

#include "detail_/slice_pimpl.hpp"
#include <tensorwrapper/buffer/replicated.hpp>
#include <tensorwrapper/buffer/replicated_view.hpp>

namespace tensorwrapper::buffer {

#define TPARAMS template<typename ReplicatedType>
#define REPLICATED_VIEW ReplicatedView<ReplicatedType>

// -----------------------------------------------------------------------------
// -- Ctors, assignment, and dtor
// -----------------------------------------------------------------------------

TPARAMS
REPLICATED_VIEW::ReplicatedView() : m_pimpl_(nullptr) {}

TPARAMS
REPLICATED_VIEW::ReplicatedView(ReplicatedType& replicated,
                                index_vector first_elem,
                                index_vector last_elem) :
  ReplicatedView(std::make_unique<detail_::SlicePIMPL<ReplicatedType>>(
    &replicated, first_elem, last_elem)) {}

TPARAMS
REPLICATED_VIEW::ReplicatedView(pimpl_pointer pimpl) :
  local_base_type(&pimpl->layout()), m_pimpl_(std::move(pimpl)) {}

TPARAMS
REPLICATED_VIEW::ReplicatedView(const ReplicatedView& other) :
  m_pimpl_(other.has_pimpl_() ? other.m_pimpl_->clone() : nullptr) {}

TPARAMS
REPLICATED_VIEW::ReplicatedView(ReplicatedView&& other) noexcept = default;

TPARAMS
auto REPLICATED_VIEW::operator=(const ReplicatedView& rhs) -> ReplicatedView& {
    if(this != &rhs) {
        m_pimpl_ = rhs.has_pimpl_() ? rhs.m_pimpl_->clone() : nullptr;
    }
    return *this;
}

TPARAMS
auto REPLICATED_VIEW::operator=(ReplicatedView&& rhs) noexcept
  -> ReplicatedView& {
    if(this != &rhs) {
        m_pimpl_ = rhs.has_pimpl_() ? rhs.m_pimpl_->clone() : nullptr;
    }
    return *this;
}

TPARAMS
REPLICATED_VIEW::~ReplicatedView() noexcept = default;

// -----------------------------------------------------------------------------
// -- Protected methods
// -----------------------------------------------------------------------------

TPARAMS
auto REPLICATED_VIEW::get_elem_(index_vector index) const
  -> const_element_reference {
    assert_pimpl_();
    return m_pimpl_->get_elem(index);
}

TPARAMS
void REPLICATED_VIEW::set_elem_(index_vector index, element_type value) {
    assert_pimpl_();
    if constexpr(std::is_const_v<ReplicatedType>) {
        throw std::runtime_error(
          "Cannot set element of a const ReplicatedView");
    } else {
        m_pimpl_->set_elem(index, value);
    }
}

// -----------------------------------------------------------------------------
// -- Private methods
// -----------------------------------------------------------------------------

TPARAMS
bool REPLICATED_VIEW::has_pimpl_() const noexcept {
    return static_cast<bool>(m_pimpl_);
}

TPARAMS
void REPLICATED_VIEW::assert_pimpl_() const {
    if(has_pimpl_()) return;
    throw std::runtime_error(
      "ReplicatedView has no PIMPL. Was it default constructed?");
}

#undef REPLICATED_VIEW
#undef TPARAMS

template class ReplicatedView<Replicated>;
template class ReplicatedView<const Replicated>;

} // namespace tensorwrapper::buffer
