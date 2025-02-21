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
#include <tensorwrapper/allocator/contiguous.hpp>
#include <tensorwrapper/buffer/buffer_fwd.hpp>
#include <tensorwrapper/types/floating_point.hpp>

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
template<typename FloatType>
class Eigen : public Contiguous<FloatType> {
private:
    /// The type of *this
    using my_type = Eigen<FloatType>;

    /// The class *this inherits from
    using my_base_type = Contiguous<FloatType>;

public:
    // Pull in base class's types
    using typename my_base_type::base_pointer;
    using typename my_base_type::buffer_base_pointer;
    using typename my_base_type::buffer_base_reference;
    using typename my_base_type::const_base_reference;
    using typename my_base_type::const_buffer_base_reference;
    using typename my_base_type::const_labeled_reference;
    using typename my_base_type::dsl_reference;
    using typename my_base_type::element_type;
    using typename my_base_type::label_type;
    using typename my_base_type::layout_pointer;
    using typename my_base_type::runtime_view_type;

    /// Type of a buffer containing an Eigen tensor
    template<unsigned int Rank>
    using eigen_buffer_type = buffer::Eigen<FloatType, Rank>;

    /// Type of a mutable reference to an object of type eigen_buffer_type
    using eigen_buffer_reference = eigen_buffer_type&;

    /// Type of a read-only reference to an object of type eigen_buffer_type
    using const_eigen_buffer_reference = const eigen_buffer_type&;

    /// Type of a pointer to an eigen_buffer_type object
    using eigen_buffer_pointer = std::unique_ptr<eigen_buffer_type>;

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

    static base_pointer make_eigen_allocator(unsigned int rank,
                                             runtime_view_type rv);

protected:
    /** @brief Polymorphic allocation of a new buffer.
     *
     *  This method overrides the polymorphic allocation so that it creates a
     *  new Eigen buffer.
     */
    buffer_base_pointer allocate_(layout_pointer playout) override;

    buffer_base_pointer construct_(layout_pointer playout,
                                   element_type value) override;

    /// Implements clone by calling copy ctor
    base_pointer clone_() const override {
        return std::make_unique<my_type>(*this);
    }

    /// Implements are_equal, by deferring to the base's operator==
    bool are_equal_(const_base_reference rhs) const noexcept override {
        return my_base_type::template are_equal_impl_<my_type>(rhs);
    }

    dsl_reference addition_assignment_(label_type this_labels,
                                       const_labeled_reference lhs,
                                       const_labeled_reference rhs) override;

    dsl_reference subtraction_assignment_(label_type this_labels,
                                          const_labeled_reference lhs,
                                          const_labeled_reference rhs) override;

    dsl_reference multiplication_assignment_(
      label_type this_labels, const_labeled_reference lhs,
      const_labeled_reference rhs) override;

    dsl_reference permute_assignment_(label_type this_labels,
                                      const_labeled_reference rhs) override;
};

// -----------------------------------------------------------------------------
// -- Explicit class template declarations
// -----------------------------------------------------------------------------

#define DECLARE_EIGEN_ALLOCATOR(TYPE) extern template class Eigen<TYPE>

TW_APPLY_FLOATING_POINT_TYPES(DECLARE_EIGEN_ALLOCATOR);

#undef DECLARE_EIGEN_ALLOCATOR

} // namespace tensorwrapper::allocator
