#include <tensorwrapper/shape/smooth_view.hpp>

namespace tensorwrapper::shape::detail_ {

// Eventually this should be turned into a base class and the current
// implementation moved into a derived class that wraps an existing Smooth
// object
template<typename SmoothType>
class SmoothViewPIMPL {
public:
    /// Type of the class *this is implementing
    using parent_type = SmoothView<SmoothType>;

    /// Pull in parent's types
    ///@{
    using smooth_pointer   = typename parent_type::smooth_traits::pointer;
    using smooth_reference = typename parent_type::smooth_reference;
    using pimpl_pointer    = typename parent_type::pimpl_pointer;
    using rank_type        = typename parent_type::rank_type;
    using size_type        = typename parent_type::size_type;
    ///@}

    explicit SmoothViewPIMPL(smooth_reference shape) : m_pshape_(&shape) {}

    pimpl_pointer clone() const {
        return std::make_unique<SmoothViewPIMPL>(*this);
    }

    rank_type extent(size_type i) const { return m_pshape_->extent(i); }

    rank_type rank() const noexcept { return m_pshape_->rank(); }

    size_type size() const noexcept { return m_pshape_->size(); }

    bool operator==(const SmoothViewPIMPL& rhs) const noexcept {
        return (*m_pshape_) == (*rhs.m_pshape_);
    }

private:
    smooth_pointer m_pshape_;
};

} // namespace tensorwrapper::shape::detail_