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
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::buffer {
namespace detail_ {
template<typename FloatType>
class EigenPIMPL;

}

/** @brief A buffer which wraps an Eigen object.
 *
 *  @tparam FloatType The type used to store the elements of the object.
 *
 *  Right now the backend is always an Eigen Tensor, but concievably it could
 *  be generalized to be matrices or Eigen's map class.
 */
template<typename FloatType>
class Eigen : public Contiguous<FloatType> {
private:
    /// Type of *this
    using my_type = Eigen<FloatType>;

    /// Type *this derives from
    using my_base_type = Contiguous<FloatType>;

public:
    /// Pull in base class's types
    using typename my_base_type::allocator_base_pointer;
    using typename my_base_type::buffer_base_pointer;
    using typename my_base_type::const_allocator_reference;
    using typename my_base_type::const_buffer_base_reference;
    using typename my_base_type::const_labeled_reference;
    using typename my_base_type::const_layout_reference;
    using typename my_base_type::const_pointer;
    using typename my_base_type::const_reference;
    using typename my_base_type::dsl_reference;
    using typename my_base_type::element_type;
    using typename my_base_type::element_vector;
    using typename my_base_type::index_vector;
    using typename my_base_type::label_type;
    using typename my_base_type::layout_pointer;
    using typename my_base_type::layout_type;
    using typename my_base_type::pointer;
    using typename my_base_type::polymorphic_base;
    using typename my_base_type::reference;
    using typename my_base_type::size_type;

    using pimpl_type            = detail_::EigenPIMPL<FloatType>;
    using pimpl_pointer         = std::unique_ptr<pimpl_type>;
    using pimpl_reference       = pimpl_type&;
    using const_pimpl_reference = const pimpl_type&;

    /** @brief Creates a buffer with no layout and a default initialized
     *         tensor.
     *
     *  @throw None No throw guarantee.
     */
    Eigen() noexcept;

    /** @brief Wraps the provided tensor.
     *
     *  @tparam DataType The type of the input tensor. Must be implicitly
     *                     convertible to an object of type data_type.
     *
     *  @param[in] t The tensor to wrap.
     *  @param[in] layout The physical layout of @p t.
     *
     *  @throw std::bad_alloc if there is a problem copying @p layout. Strong
     *                        throw guarantee.
     */
    Eigen(pimpl_pointer pimpl, const_layout_reference layout,
          const_allocator_reference allocator) :
      Eigen(std::move(pimpl), layout.template clone_as<layout_type>(),
            allocator.clone()) {}

    Eigen(pimpl_pointer pimpl, layout_pointer playout,
          allocator_base_pointer pallocator);

    /** @brief Initializes *this with a copy of @p other.
     *
     *  @param[in] other The object to copy.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    Eigen(const Eigen& other);

    /** @brief Initializes *this with the state from @p other.
     *
     *  @param[in,out] other The object to take the state from. After this call
     *                       @p other will be in a valid, but otherwise
     *                       undefined state.
     *
     *  @throw None No throw guarantee.
     */
    Eigen(Eigen&& other) noexcept;

    /** @brief Replaces the state in *this with a copy of the state in @p rhs.
     *
     *  @param[in] rhs The object to copy the state from.
     *
     *  @return *this after replacing its state with a copy of @p rhs.
     *
     *  @throw std::bad_alloc if the copy fails to allocate memory. Strong
     *                        throw guarantee.
     */
    Eigen& operator=(const Eigen& rhs);

    /** @brief Replaces the state in *this with the state in @p rhs.
     *
     *  @param[in,out] rhs The Eigen object to take the state from. After this
     *                     method is called @p rhs will be in a valid, but
     *                     otherwise undefined state.
     *
     *  @return *this after taking the state from @p rhs.
     *
     *  @throw None No throw guarantee.
     */
    Eigen& operator=(Eigen&& rhs) noexcept;

    /// Defaulted no throw dtor
    ~Eigen() noexcept;

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Exchanges the contents of *this with @p other.
     *
     *  @param[in,out] other The buffer to swap state with.
     *
     *  @throw None No throw guarantee.
     */
    void swap(Eigen& other) noexcept;

    /** @brief Is *this value equal to @p rhs?
     *
     *  Two Eigen objects are value equal if they both have the same layout and
     *  they both have the same values.
     *
     *  @note For tensors where the @p FloatType is an uncertain floating point
     *  number, the tensors are required to have the same sources of
     *  uncertainty.
     *
     *  @param[in] rhs The object to compare against.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const Eigen& rhs) const noexcept;

    /** @brief Is *this different from @p rhs?
     *
     *  This class defines different as not value equal. See operator== for the
     *  definition of value equal.
     *
     *  @param[in] rhs The object to compare *this to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const Eigen& rhs) const noexcept { return !(*this == rhs); }

protected:
    /// Implements clone by calling copy ctor
    buffer_base_pointer clone_() const override;

    /// Implements are_equal by calling are_equal_impl_
    bool are_equal_(const_buffer_base_reference rhs) const noexcept override;

    /// Implements addition_assignment by calling addition_assignment on state
    dsl_reference addition_assignment_(label_type this_labels,
                                       const_labeled_reference lhs,
                                       const_labeled_reference rhs) override;

    /// Calls subtraction_assignment on each member
    dsl_reference subtraction_assignment_(label_type this_labels,
                                          const_labeled_reference lhs,
                                          const_labeled_reference rhs) override;

    /// Calls multiplication_assignment on each member
    dsl_reference multiplication_assignment_(
      label_type this_labels, const_labeled_reference lhs,
      const_labeled_reference rhs) override;

    /// Calls permute_assignment on each member
    dsl_reference permute_assignment_(label_type this_labels,
                                      const_labeled_reference rhs) override;

    /// Scales *this by @p scalar
    dsl_reference scalar_multiplication_(label_type this_labels, double scalar,
                                         const_labeled_reference rhs) override;

    /// Implements getting the raw pointer
    pointer get_mutable_data_() noexcept override;

    /// Implements getting the raw pointer (read-only)
    const_pointer get_immutable_data_() const noexcept override;

    /// Implements read-only element access
    const_reference get_elem_(index_vector index) const override;

    // Implements element updating
    void set_elem_(index_vector index, element_type new_value) override;

    /// Implements read-only element access by ordinal index
    const_reference get_data_(size_type index) const override;

    // Implements element updating by ordinal index
    void set_data_(size_type index, element_type new_value) override;

    /// Implements filling the tensor
    void fill_(element_type value) override;

    /// Implements copying new values into the tensor
    void copy_(const element_vector& values) override;

    /// Implements to_string
    typename polymorphic_base::string_type to_string_() const override;

    /// Implements add_to_stream
    std::ostream& add_to_stream_(std::ostream& os) const override;

private:
    /// True if *this has a PIMPL
    bool has_pimpl_() const noexcept;

    /// Throws std::runtime_error if *this has no PIMPL
    void assert_pimpl_() const;

    /// Asserts *this has a PIMPL then returns it
    pimpl_reference pimpl_();

    /// Assert *this has a PIMPL then returns it
    const_pimpl_reference pimpl_() const;

    /// The object actually implementing *this
    pimpl_pointer m_pimpl_;
};

/** @brief Wraps downcasting a buffer to an Eigen buffer.
 *
 *  @tparam FloatType The type of the elements in the resulting Buffer.
 *
 *  This function is a convience function for using an allocator to convert
 *  @p b to a buffer::Eigen object.
 *
 *  @param[in] b The BufferBase object to convert.
 *
 *  @return A reference to @p b after downcasting it.
 */
template<typename FloatType>
Eigen<FloatType>& to_eigen_buffer(BufferBase& b);

/** @brief Wraps downcasting a buffer to an Eigen buffer.
 *
 *  @tparam FloatType The type of the elements in the resulting Buffer.
 *
 *  This function is the same as the non-const overload except that result will
 *  be read-only.
 *
 *  @param[in] b The BufferBase object to convert.
 *
 *  @return A reference to @p b after downcasting it.
 */
template<typename FloatType>
const Eigen<FloatType>& to_eigen_buffer(const BufferBase& b);

#define DECLARE_EIGEN_BUFFER(TYPE) extern template class Eigen<TYPE>
#define DECLARE_TO_EIGEN_BUFFER(TYPE) \
    extern template Eigen<TYPE>& to_eigen_buffer(BufferBase&)
#define DECLARE_TO_CONST_EIGEN_BUFFER(TYPE) \
    extern template const Eigen<TYPE>& to_eigen_buffer(const BufferBase&)

TW_APPLY_FLOATING_POINT_TYPES(DECLARE_EIGEN_BUFFER);
TW_APPLY_FLOATING_POINT_TYPES(DECLARE_TO_EIGEN_BUFFER);
TW_APPLY_FLOATING_POINT_TYPES(DECLARE_TO_CONST_EIGEN_BUFFER);

#undef DECLARE_EIGEN_BUFFER
#undef DECLARE_TO_EIGEN_BUFFER
#undef DECLARE_TO_CONST_EIGEN_BUFFER

} // namespace tensorwrapper::buffer
