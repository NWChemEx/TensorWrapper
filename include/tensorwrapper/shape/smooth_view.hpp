#pragma once
#include <tensorwrapper/shape/shape_traits.hpp>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::shape {

template<typename SmoothType>
class SmoothView {
private:
    /// Type of *this
    using my_type = SmoothView<SmoothType>;

    /// Type defining the traits for *this
    using traits_type = ShapeTraits<my_type>;

public:
    /// Types needed to implement Smooth's interface
    ///@{
    using smooth_traits          = typename traits_type::smooth_traits;
    using smooth_type            = typename smooth_traits::value_type;
    using smooth_reference       = typename smooth_traits::reference;
    using const_smooth_reference = typename smooth_traits::const_reference;
    using rank_type              = typename smooth_traits::rank_type;
    using size_type              = typename smooth_traits::size_type;
    ///@}

    using pimpl_pointer = typename traits_type::pimpl_pointer;

    SmoothView(smooth_reference smooth);

    SmoothView(const SmoothView& other);
    SmoothView(SmoothView&& other) noexcept;
    SmoothView& operator=(const SmoothView& rhs);
    SmoothView& operator=(SmoothView&& rhs) noexcept;
    ~SmoothView() noexcept;

    rank_type extent(size_type i) const;
    rank_type rank() const noexcept;
    size_type size() const noexcept;

    void swap(SmoothView& rhs) noexcept;

    bool operator==(const SmoothView& rhs) const noexcept;
    bool operator!=(const SmoothView& rhs) const noexcept {
        return !((*this) == rhs);
    }

private:
    bool has_pimpl_() const noexcept;
    pimpl_pointer clone_() const;
    pimpl_pointer m_pimpl_;
};

extern template class SmoothView<Smooth>;
extern template class SmoothView<const Smooth>;

} // namespace tensorwrapper::shape