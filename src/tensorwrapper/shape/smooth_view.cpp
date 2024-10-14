#include "detail_/smooth_view_pimpl.hpp"
#include <tensorwrapper/shape/smooth_view.hpp>

namespace tensorwrapper::shape {

#define TPARAMS template<typename SmoothType>
#define SMOOTH_VIEW SmoothView<SmoothType>

TPARAMS
SMOOTH_VIEW::SmoothView(smooth_reference smooth) :
  m_pimpl_(std::make_unique<typename traits_type::pimpl_type>(smooth)) {}

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
bool SMOOTH_VIEW::operator==(const SmoothView& rhs) const noexcept {
    if(has_pimpl_() != rhs.has_pimpl_()) return false;
    if(!has_pimpl_()) return true;
    return (*m_pimpl_) == (*rhs.m_pimpl_);
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

template class SmoothView<Smooth>;
template class SmoothView<const Smooth>;

} // namespace tensorwrapper::shape