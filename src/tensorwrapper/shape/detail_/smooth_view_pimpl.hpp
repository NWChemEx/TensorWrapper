#pragma once
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/shape/smooth_view.hpp>

namespace tensorwrapper::shape::detail_ {

/** @brief Defines the API for all SmoothView PIMPLs.
 *
 *  The data for a SmoothView can be laid out in a number of different ways.
 *  This class defines the API for accessing it.
 *
 *  @tparam SmoothType The type *this will be a view of.
 */
template<typename SmoothType>
class SmoothViewPIMPL : public tensorwrapper::detail_::PolymorphicBase<
                          SmoothViewPIMPL<SmoothType>> {
private:
    /// Type of *this
    using my_type = SmoothViewPIMPL<SmoothType>;

    /// Type *this actually inherits from.
    using my_base = tensorwrapper::detail_::PolymorphicBase<my_type>;

public:
    /// Type of the class *this is implementing
    using parent_type = SmoothView<SmoothType>;

    /// Pull in parent's types
    ///@{
    using rank_type = typename parent_type::rank_type;
    using size_type = typename parent_type::size_type;
    ///@}

    /// Pull in base's types
    using const_base_reference = typename my_base::const_base_reference;

    /// Type of a SmoothViewPIMPL if it aliases a const Smooth
    using const_smooth_view_pimpl_pointer =
      typename ShapeTraits<parent_type>::const_pimpl_pointer;

    /// Derived class implements by overriding extent_
    rank_type extent(size_type i) const { return extent_(i); }

    /// Derived class implements by overriding rank_
    rank_type rank() const noexcept { return rank_(); }

    /// Derived class implements by overriding size_
    size_type size() const noexcept { return size_(); }

    /// Derived class implements by overriding as_const_()
    const_smooth_view_pimpl_pointer as_const() const { return as_const_(); }

protected:
    /// Derived class should implement to be consistent with SmoothView::extent
    virtual rank_type extent_(size_type i) const = 0;

    /// Derived class should implement to be consistent with SmoothView::rank
    virtual rank_type rank_() const noexcept = 0;

    /// Derived class should implement to be consistent with SmoothView::size
    virtual size_type size_() const noexcept = 0;

    /// Used to create a PIMPL for SmoothView<const T>
    virtual const_smooth_view_pimpl_pointer as_const_() const = 0;

    /// Compares state through common API of this class
    bool are_equal_(const_base_reference rhs) const noexcept override {
        if(rank() != rhs.rank()) return false;
        for(size_type i = 0; i < rank(); ++i)
            if(extent(i) != rhs.extent(i)) return false;
        return true;
    }
};

} // namespace tensorwrapper::shape::detail_