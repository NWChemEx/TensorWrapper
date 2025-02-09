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
#include <tensorwrapper/allocator/replicated.hpp>
#include <tensorwrapper/buffer/buffer_fwd.hpp>

#ifdef ENABLE_SIGMA
#include <sigma/sigma.hpp>
#endif

namespace tensorwrapper::allocator {

/** @brief Used to allocate buffers which rely on Eigen tensors.
 *
 *  @tparam FloatType The numerical type the buffer will use to store the
 *                    elements.
 *  @tparam Rank The rank of the tensor stored in the buffer.
 *
 *  This allocator is capable of creating buffers with Eigen tensors in them.
 *
 */
template<typename FloatType, unsigned short Rank>
class Eigen : public Replicated {
private:
    /// The type of *this
    using my_type = Eigen<FloatType, Rank>;

    /// The class *this inherits from
    using my_base_type = Replicated;

public:
    // Pull in base class's types
    using my_base_type::base_pointer;
    using my_base_type::buffer_base_pointer;
    using my_base_type::buffer_base_reference;
    using my_base_type::const_base_reference;
    using my_base_type::const_buffer_base_reference;
    using my_base_type::layout_pointer;
    using my_base_type::runtime_view_type;

    /// Type of a buffer containing an Eigen tensor
    using eigen_buffer_type = buffer::Eigen<FloatType, Rank>;

    /// Type of a mutable reference to an object of type eigen_buffer_type
    using eigen_buffer_reference = eigen_buffer_type&;

    /// Type of a read-only reference to an object of type eigen_buffer_type
    using const_eigen_buffer_reference = const eigen_buffer_type&;

    /// Type of a pointer to an eigen_buffer_type object
    using eigen_buffer_pointer = std::unique_ptr<eigen_buffer_type>;

    /// Type of a layout which can be used to create an Eigen tensor
    using eigen_layout_type = layout::Physical;

    /// Type of a read-only reference to an object of type eigen_layout_type
    using const_eigen_layout_reference = const eigen_layout_type&;

    /// Type of a pointer to an eigen_layout_type object
    using eigen_layout_pointer = std::unique_ptr<eigen_layout_type>;

    // Reuse base class's ctors
    using my_base_type::my_base_type;

    // -------------------------------------------------------------------------
    // -- Ctor
    // -------------------------------------------------------------------------

    /** @brief Creates a new Eigen allocator tied to the runtime @p rv.
     *
     *  This ctor simply dispatches to the base class's ctor with the same
     *  signature. See the base class's description for more detail.
     *
     *  @param[in] rv The runtime to use for allocating.
     *
     *  @throw None No throw guarantee.
     */
    explicit Eigen(runtime_view_type rv) : my_base_type(std::move(rv)) {}

    /** @brief Copies @p layout and dispatches to other overload.
     *
     *  The buffer resulting from an allocator owns its layout. This method is
     *  a convenience function for when the layout for the buffer has not been
     *  allocated yet.
     *
     *  @param[in] layout The to copy for the resulting buffer.
     *
     *  @return An uninitialized buffer containing a copy of @p layout.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy or the
     *                        resulting buffer. Strong throw guarantee.
     *  @throw std::runtime_error if the provided layout is not compatible with
     *                            *this. See the primary method for more
     *                            details.
     */
    eigen_buffer_pointer allocate(const_eigen_layout_reference layout) {
        return allocate(std::make_unique<eigen_layout_type>(layout));
    }

    /** @brief Primary method for allocating Eigen-based buffers.
     *
     *  This method simply checks that @p playout points to a layout compatible
     *  with the template parameters of *this and then creates a new Eigen
     *  tensor object. The elements of the Eigen tensor object are NOT
     *  initialized.
     *
     *  @param[in] playout A pointer to the layout for the new buffer.
     *
     *  @return An uninitialized buffer containing @p playout.
     *
     *  @throw std::runtime_error if the provided layout does not have the same
     *                            rank as @p Rank. Strong throw guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the new buffer.
     *                        Strong throw guarantee.
     */
    eigen_buffer_pointer allocate(eigen_layout_pointer playout);

    /** @brief Allocates and initializes an Eigen buffer.
     *
     *  @tparam LayoutType The type of @p layout. Must be a type such that
     *                     `allocate(layout)` is a valid call.
     *
     *  @param[in] layout The layout for the buffer.
     *  @param[in] value  The value to initialize the tensor with.
     *
     *  @return A buffer which is allocated and initialized.
     *
     *  @throw std::runtime_error if the provided layout is not compatible with
     *                            the template parameters of *this. Strong throw
     *                            guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the buffer.
     *                        Strong throw guarantee.
     */
    template<typename LayoutType>
    eigen_buffer_pointer construct(LayoutType&& layout, FloatType value) {
        auto pbuffer = allocate(std::forward<LayoutType>(layout));
        pbuffer->value().setConstant(value);
        return pbuffer;
    }

    /** @brief Determines if @p buffer can be rebound as an Eigen buffer.
     *
     *  Rebinding a buffer allows the same memory to be viewed as a (possibly)
     *  different type of buffer.
     *
     *  @param[in] buffer The tensor we are attempting to rebind.
     *
     *  @return True if @p buffer can be rebound to the type of buffer
     *          associated with this allocator and false otherwise.
     *
     *  @throw None No throw guarantee
     */
    static bool can_rebind(const_buffer_base_reference buffer);

    /** @brief Rebinds a buffer to the same type as *this.
     *
     *  This method will convert @p buffer into a buffer which could have been
     *  allocated by *this. If @p buffer was allocated as such a buffer already,
     *  then this method is simply a downcast.
     *
     *  @param[in] buffer The buffer to rebind.
     *
     *  @return A mutable reference to @p buffer viewed as a buffer that could
     *          have been allocated by *this.
     *
     *  @throw std::runtime_error if can_rebind(buffer) is false. Strong throw
     *                            guarantee.
     */
    static eigen_buffer_reference rebind(buffer_base_reference buffer);

    /** @brief Rebinds a buffer to the same type as *this.
     *
     *  This method is the same as the non-const version except that the result
     *  is read-only. See the description for the non-const version for more
     *  details.
     *
     *  @param[in] buffer The buffer to rebind.
     *
     *  @return A read-only reference to @p buffer viewed as if it was
     *          allocated by *this.
     *
     *  @throw std::runtime_error if can_rebind(buffer) is false. Strong throw
     *                            guarantee.
     */
    static const_eigen_buffer_reference rebind(
      const_buffer_base_reference buffer);

    /** @brief Is *this value equal to @p rhs?
     *
     *  @tparam FloatType2 The numerical type @p rhs uses for its elements.
     *  @tparam Rank2 The rank of the tensors allocated by @p rhs.
     *
     *  In addition to the definition of value equal stemming from the
     *  Replicated base class, two Eigen allocators are only value equal if they
     *  produce tensors with the same rank and numerical type.
     *
     *  @param[in] rhs The allocator to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    template<typename FloatType2, unsigned short Rank2>
    bool operator==(const Eigen<FloatType2, Rank2>& rhs) const noexcept {
        if constexpr(!std::is_same_v<FloatType, FloatType2> || Rank != Rank2) {
            return false;
        } else {
            return base_type::operator==(rhs);
        }
    }

    /** @brief Is this allocator different from @p rhs?
     *
     *  @tparam FloatType2 The type @p rhs uses for floating-point elements.
     *  @tparam Rank2 The rank of @p rhs
     *
     *  This method defines "different" as "not value equal." See the
     *  documentation for operator== for the definition of value equal.
     *
     *  @param[in] rhs The allocator to compare against.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    template<typename FloatType2, unsigned short Rank2>
    bool operator!=(const Eigen<FloatType2, Rank2>& rhs) const noexcept {
        return !(*this == rhs);
    }

    static base_pointer make_eigen_allocator(unsigned int rank,
                                             runtime_view_type rv);

protected:
    /** @brief Polymorphic allocation of a new buffer.
     *
     *  This method overrides the polymorphic allocation so that it creates a
     *  new Eigen buffer.
     */
    buffer_base_pointer allocate_(layout_pointer playout) override;

    /// Implements clone by calling copy ctor
    base_pointer clone_() const override {
        return std::make_unique<my_type>(*this);
    }

    /// Implements are_equal, by deferring to the base's operator==
    bool are_equal_(const_base_reference rhs) const noexcept override {
        return my_base_type::are_equal_impl_<my_type>(rhs);
    }
};

// -----------------------------------------------------------------------------
// -- Explicit class template declarations
// -----------------------------------------------------------------------------

#define DECLARE_EIGEN_ALLOCATOR(TYPE)     \
    extern template class Eigen<TYPE, 0>; \
    extern template class Eigen<TYPE, 1>; \
    extern template class Eigen<TYPE, 2>; \
    extern template class Eigen<TYPE, 3>; \
    extern template class Eigen<TYPE, 4>; \
    extern template class Eigen<TYPE, 5>; \
    extern template class Eigen<TYPE, 6>; \
    extern template class Eigen<TYPE, 7>; \
    extern template class Eigen<TYPE, 8>; \
    extern template class Eigen<TYPE, 9>; \
    extern template class Eigen<TYPE, 10>

DECLARE_EIGEN_ALLOCATOR(float);
DECLARE_EIGEN_ALLOCATOR(double);

#ifdef ENABLE_SIGMA
DECLARE_EIGEN_ALLOCATOR(sigma::UFloat);
DECLARE_EIGEN_ALLOCATOR(sigma::UDouble);
#endif

#undef DECLARE_EIGEN_ALLOCATOR

} // namespace tensorwrapper::allocator
