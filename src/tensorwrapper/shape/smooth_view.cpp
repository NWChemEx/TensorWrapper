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

#include "detail_/smooth_alias.hpp"
#include <tensorwrapper/shape/smooth_view.hpp>

namespace tensorwrapper::shape {

#define TPARAMS template<typename SmoothType>
#define SMOOTH_VIEW SmoothView<SmoothType>

TPARAMS
SMOOTH_VIEW::SmoothView(smooth_reference smooth) :
  m_pimpl_(std::make_unique<detail_::SmoothAlias<SmoothType>>(smooth)) {}

TPARAMS
template<typename SmoothType2, typename>
SMOOTH_VIEW::SmoothView(const SmoothView<SmoothType2>& other) :
  m_pimpl_(other.m_pimpl_->as_const()) {}

TPARAMS
SMOOTH_VIEW::SmoothView(const SmoothView& other) : m_pimpl_(other.clone_()) {}

TPARAMS
SMOOTH_VIEW::SmoothView(SmoothView&& other) noexcept = default;

TPARAMS
SMOOTH_VIEW& SMOOTH_VIEW::operator=(const SmoothView& rhs) {
    if(this != &rhs) SmoothView(rhs).swap(*this);
    return *this;
}

TPARAMS
SMOOTH_VIEW& SMOOTH_VIEW::operator=(SmoothView&& rhs) noexcept = default;

TPARAMS
SMOOTH_VIEW::~SmoothView() noexcept = default;

TPARAMS
typename SMOOTH_VIEW::rank_type SMOOTH_VIEW::extent(size_type i) const {
    return m_pimpl_->extent(i);
}

TPARAMS
typename SMOOTH_VIEW::rank_type SMOOTH_VIEW::rank() const noexcept {
    return m_pimpl_->rank();
}

TPARAMS
typename SMOOTH_VIEW::size_type SMOOTH_VIEW::size() const noexcept {
    return m_pimpl_->size();
}

TPARAMS
void SMOOTH_VIEW::swap(SmoothView& rhs) noexcept {
    m_pimpl_.swap(rhs.m_pimpl_);
}

TPARAMS
bool SMOOTH_VIEW::operator==(const SmoothView<SmoothType>& rhs) const noexcept {
    if(has_pimpl_() != rhs.has_pimpl_()) return false;
    if(!has_pimpl_()) return true;
    return m_pimpl_->are_equal(*rhs.m_pimpl_);
}

TPARAMS
bool SMOOTH_VIEW::has_pimpl_() const noexcept {
    return static_cast<bool>(m_pimpl_);
}

TPARAMS
typename SMOOTH_VIEW::pimpl_pointer SMOOTH_VIEW::clone_() const {
    return has_pimpl_() ? m_pimpl_->clone() : nullptr;
}

#undef SMOOTH_VIEW
#undef TPARAMS

template SmoothView<const Smooth>::SmoothView(const SmoothView<Smooth>&);
template class SmoothView<Smooth>;
template class SmoothView<const Smooth>;

} // namespace tensorwrapper::shape
