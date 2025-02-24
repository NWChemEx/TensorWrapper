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
#include <tensorwrapper/backends/eigen.hpp>
#include <tensorwrapper/buffer/contiguous.hpp>

#ifdef ENABLE_SIGMA
#include <sigma/sigma.hpp>
#endif

namespace tensorwrapper::buffer {

/** @brief A buffer which wraps an Eigen tensor.
 *
 *  @tparam FloatType The type used to store the elements of the tensor.
 *  @tparam Rank The rank of the tensor.
 *
 */
template<typename FloatType, unsigned short Rank>
class Eigen : public Contiguous<FloatType> {
private:
    /// Type of *this
    using my_type = Eigen<FloatType, Rank>;

    /// Type *this derives from
    using my_base_type = Contiguous<FloatType>;

public:
    /// Pull in base class's types
    using typename my_base_type::buffer_base_pointer;
    using typename my_base_type::const_buffer_base_reference;
    using typename my_base_type::const_labeled_reference;
    using typename my_base_type::const_layout_reference;
    using typename my_base_type::const_pointer;
    using typename my_base_type::dsl_reference;
    using typename my_base_type::label_type;
    using typename my_base_type::pointer;
    using typename my_base_type::polymorphic_base;

    /// Type of a rank @p Rank tensor using floats of type @p FloatType
    using data_type = eigen::data_type<FloatType, Rank>;

    /// Mutable reference to an object of type data_type
    using data_reference = data_type&;

    /// Read-only reference to an object of type data_type
    using const_data_reference = const data_type&;

    /** @brief Creates a buffer with no layout and a default initialized
     *         tensor.
     *
     *  @throw None No throw guarantee.
     */
    Eigen() noexcept = default;

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
    template<typename DataType>
    Eigen(DataType&& t, const_layout_reference layout) :
      my_base_type(layout), m_tensor_(std::forward<DataType>(t)) {}

    /** @brief Initializes *this with a copy of @p other.
     *
     *  @param[in] other The object to copy.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    Eigen(const Eigen& other) = default;

    /** @brief Initializes *this with the state from @p other.
     *
     *  @param[in,out] other The object to take the state from. After this call
     *                       @p other will be in a valid, but otherwise
     *                       undefined state.
     *
     *  @throw None No throw guarantee.
     */
    Eigen(Eigen&& other) = default;

    /** @brief Replaces the state in *this with a copy of the state in @p rhs.
     *
     *  @param[in] rhs The object to copy the state from.
     *
     *  @return *this after replacing its state with a copy of @p rhs.
     *
     *  @throw std::bad_alloc if the copy fails to allocate memory. Strong
     *                        throw guarantee.
     */
    Eigen& operator=(const Eigen& rhs) = default;

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
    Eigen& operator=(Eigen&& rhs) = default;

    // -------------------------------------------------------------------------
    // -- State accessors
    // -------------------------------------------------------------------------

    /** @brief Allows direct access to the wrapped tensor.
     *
     *
     *  @return A mutable reference to the wrapped tensor.
     *
     *  @throw None No throw guarantee
     */
    auto& value() { return m_tensor_; }

    /** @brief Provides direct access to the wrapped tensor.
     *
     *  @return A read-only reference to the wrapped tensor.
     *
     *  @throw None No throw guarantee.
     */
    const auto& value() const { return m_tensor_; }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

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
    bool operator==(const Eigen& rhs) const noexcept {
        if(my_base_type::operator!=(rhs)) return false;
        eigen::data_type<FloatType, 0> r = (m_tensor_ - rhs.m_tensor_).sum();
        return r() == FloatType(0.0);
    }

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
    buffer_base_pointer clone_() const override {
        return std::make_unique<my_type>(*this);
    }

    /// Implements are_equal by calling are_equal_impl_
    bool are_equal_(const_buffer_base_reference rhs) const noexcept override {
        return my_base_type::template are_equal_impl_<my_type>(rhs);
    }

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

    dsl_reference scalar_multiplication_(label_type this_labels, double scalar,
                                         const_labeled_reference rhs) override;

    pointer data_() noexcept override { return m_tensor_.data(); }

    const_pointer data_() const noexcept override { return m_tensor_.data(); }

    /// Implements to_string
    typename polymorphic_base::string_type to_string_() const override;

private:
    dsl_reference hadamard_(label_type this_labels, const_labeled_reference lhs,
                            const_labeled_reference rhs);

    dsl_reference contraction_(label_type this_labels,
                               const_labeled_reference lhs,
                               const_labeled_reference rhs);

    /// The actual Eigen tensor
    data_type m_tensor_;
};

#define DECLARE_EIGEN_BUFFER(TYPE)        \
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

DECLARE_EIGEN_BUFFER(float);
DECLARE_EIGEN_BUFFER(double);

#ifdef ENABLE_SIGMA
DECLARE_EIGEN_BUFFER(sigma::UFloat);
DECLARE_EIGEN_BUFFER(sigma::UDouble);
#endif

#undef DECLARE_EIGEN_BUFFER

} // namespace tensorwrapper::buffer
