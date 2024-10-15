#pragma once
#include <tensorwrapper/detail_/view_traits.hpp>
#include <tensorwrapper/shape/shape_traits.hpp>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::shape {

/** @brief Wraps existing state in an API compatible with SmoothView.
 *
 *  @tparam SmoothType Type of Smooth object *this is acting as. Expected to be
 *                     either Smooth or const Smooth.
 *
 *  Sometimes we have state which may not actually be in a Smooth object, but
 *  is capable of being used as a Smooth object. This class maps the Smooth
 *  API to the existing state.
 */
template<typename SmoothType>
class SmoothView {
private:
    /// Type of *this
    using my_type = SmoothView<SmoothType>;

    /// Type defining the traits for *this
    using traits_type = ShapeTraits<my_type>;

    /// Bind SmoothType for
    template<typename T>
    using enable_if_mutable_to_immutable_cast_t =
      tensorwrapper::detail_::enable_if_mutable_to_immutable_cast_t<T,
                                                                    SmoothType>;

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

    /** @brief Creates a view of an existing Smooth object.
     *
     *  In order to treat SmoothView objects on the same footing as Smooth
     *  objects it must be possible to implicitly convert between the two.
     *  This ctor will implicitly convert @p smooth into a SmoothView object.
     *
     *  @param[in] smooth The object to convert.
     *
     *  @throw std::bad_alloc if there is a problem allocating the PIMPL.
     *                        Strong throw guarantee.
     */
    SmoothView(smooth_reference smooth);

    /** @brief Implicitly converts mutable views into read-only views.
     *
     *  @tparam SmoothType2 The type @p other is a view of. This method will
     *                      only participate in overload resolution if
     *                      SmoothType2 is `const Smooth`.
     *  @tparam <Anonymous> Type parameter used to disable this method when
     *                      SmoothType2 is not `const Smooth` and/or when
     *                      SmoothType is not `Smooth`.
     *
     *  Views act like references to an object. Views of mutable objects should
     *  be usable wherever views to read-only objects are used. This ctor
     *  enables the implicit conversion from mutable view to read-only view in
     *  order to make that possible.
     *
     *  @param[in] other The view to convert to a read-only view.
     *
     *  @throw std::bad_alloc if there is a problem allocating the PIMPL. Strong
     *                        throw guarantee.
     */
    template<typename SmoothType2,
             typename = enable_if_mutable_to_immutable_cast_t<SmoothType2>>
    SmoothView(const SmoothView<SmoothType2>& other);

    SmoothView(const SmoothView& other);
    SmoothView(SmoothView&& other) noexcept;
    SmoothView& operator=(const SmoothView& rhs);
    SmoothView& operator=(SmoothView&& rhs) noexcept;

    /// Nothrow defaulted dtor
    ~SmoothView() noexcept;

    rank_type extent(size_type i) const;
    rank_type rank() const noexcept;
    size_type size() const noexcept;

    /// Swaps the state of *this with that of @p rhs
    void swap(SmoothView& rhs) noexcept;

    bool operator==(const SmoothView<const SmoothType>& rhs) const noexcept;

    /** @brief Is *this different from @p rhs?
     *
     *  @tparam SmoothType2 The type @p rhs is a view of. Expected to be Smooth
     *                      or const Smooth.
     *
     *  This method defines "different" as not value equal. See operator== for
     *  the definition of value equal.
     *
     *  @param[in] rhs The view to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    template<typename SmoothType2>
    bool operator!=(const SmoothView<SmoothType2>& rhs) const noexcept {
        return !((*this) == rhs);
    }

protected:
    /// Lets the class access PIMPLs regardless of template type parameter
    template<typename SmoothType2>
    friend class SmoothView;

private:
    /// Type of a pointer to the PIMPL
    using pimpl_pointer = typename traits_type::pimpl_pointer;

    /// Does *this have a PIMPL?
    bool has_pimpl_() const noexcept;

    /// Makes a deep copy of the PIMPL
    pimpl_pointer clone_() const;

    /// The object implementing *this
    pimpl_pointer m_pimpl_;
};

extern template class SmoothView<Smooth>;
extern template class SmoothView<const Smooth>;

} // namespace tensorwrapper::shape