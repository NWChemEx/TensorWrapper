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

#pragma once
#include <tensorwrapper/detail_/view_traits.hpp>
#include <tensorwrapper/shape/shape_traits.hpp>

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

    /** @brief Creates a new view aliasing the same Smooth object as @p other.
     *
     *  Views alias their state. The view constructed by this copy ctor will
     *  alias the same state that is aliased by @p other. In this sense it is
     *  a shallow copy of the aliased state and a deep copy of @p other.
     *
     *  @param[in] other The view to copy.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    SmoothView(const SmoothView& other);

    /** @brief Creates a new view by taking the state of @p other.
     *
     *  This ctor initializes *this by taking the state from @p other. After
     *  construction *this will alias the same object @p other did. It is worth
     *  noting the aliased object is untouched after this operation.
     *
     *  @param[in,out] other The object to take the state from. After this
     *                       operation @p other will be in a valid, but
     *                       otherwise undefined state.
     *
     *  @throw None No throw guarantee.
     */
    SmoothView(SmoothView&& other) noexcept;

    /** @brief Overwrites *this to alias the same Smooth object as @p other.
     *
     *  This operator causes the state in *this to instead alias the Smooth
     *  object in @p other. This does not release the state associated with the
     *  aliased object.
     *
     *  @param[in] other The view to copy.
     *
     *  @return *this after making it alias the state in @p other.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    SmoothView& operator=(const SmoothView& rhs);

    /** @brief Overrides the state of *this with the state of @p other.
     *
     *  This operator causes the state to be replaced by the state in @p other.
     *  This does not release the state associated with the aliased object nor
     *  does it take state from the aliased object.
     *
     *  @param[in,out] other The object to take the state from. After this
     *                       operation @p other will be in a valid, but
     *                       otherwise undefined state.
     *
     *  @return *this after taking the state of @p other.
     *
     *  @throw None No throw guarantee.
     */
    SmoothView& operator=(SmoothView&& rhs) noexcept;

    /// Nothrow defaulted dtor
    ~SmoothView() noexcept;

    /** @brief What is the extent of the i-th mode of the tensor with the
     *         aliased shape?
     *
     *  @param[in] i The offset of the requested mode. @p i must be in the
     *               range [0, size()).
     *
     *  @return The length of the @p i-th mode in a tensor with the aliased
     *          shape.
     *
     *  @throw std::out_of_range if @p i is not in the range [0, size()). Strong
     *                           throw guarantee.
     */
    rank_type extent(size_type i) const;

    /** @brief What is the rank of the tensor the aliased shape describes?
     *
     *  @return The rank of the tensor with the aliased shape.
     *
     *  @throw None No throw guarantee.
     */
    rank_type rank() const noexcept;

    /** @brief How many elements are in the tensor the aliased shape describes?
     *
     *  @return The number of elements in a tensor with the aliased shape.
     *
     *  @throw None No throw guarantee.
     */
    size_type size() const noexcept;

    /// Swaps the state of *this with that of @p rhs
    void swap(SmoothView& rhs) noexcept;

    /** @brief Is the Smooth shape aliased by *this the same as that aliased by
     *         @p rhs?
     *
     *  Two SmoothView objects are value equal if the Smooth objects they alias
     *  compare value equal.
     *
     *  @param[in] rhs The view aliasing the shape to compare to.
     *
     *  @return True if *this aliases a Smooth object which is value equal to
     *          that aliased by @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const SmoothView& rhs) const noexcept;

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
