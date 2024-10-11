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
#include <tensorwrapper/tensor/detail_/tensor_input.hpp>

namespace tensorwrapper {
namespace detail_ {
class TensorPIMPL;
}

/** @brief Represents a multi-dimensional array of values.
 *
 *  The Tensor class is envisioned as being the most user-facing class of
 *  TensorWrapper and forms the entry point into TensorWrapper's DSL.
 */
class Tensor {
private:
    /// Type of a helper class which collects the inputs needed to make a tensor
    using input_type = detail_::TensorInput;

    template<typename T>
    using disable_if_tensor_t = std::enable_if_t<!std::is_same_v<T, Tensor>, T>;

public:
    /// Type of the object implementing *this
    using pimpl_type = detail_::TensorPIMPL;

    /// Type of a pointer to an object of type pimpl_type
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    /// Type of an object storing the logical layout of the tensor
    using logical_layout_type = input_type::logical_layout_type;

    /// Type of a read-only reference to the tensor's logical layout
    using const_logical_reference = input_type::const_logical_reference;

    /// Type of a pointer to the tensor's logical layout
    using logical_layout_pointer = input_type::logical_layout_pointer;

    /// Type of a read-only reference to the tensor's buffer
    using const_buffer_reference = input_type::const_buffer_reference;

    /// Type of a pointer to the tensor's buffer
    using buffer_pointer = input_type::buffer_pointer;

    /// Type of an initializer list if *this is a scalar
    using scalar_il_type = double;

    /// Type of an initializer list if *this is a vector
    using vector_il_type = std::initializer_list<scalar_il_type>;

    /// Type of an initializer list if *this is a matrix
    using matrix_il_type = std::initializer_list<vector_il_type>;

    /// Type of an initializer list if *this is a rank 3 tensor
    using tensor3_il_type = std::initializer_list<matrix_il_type>;

    /// Type of an initializer list if *this is a rank 4 tensor
    using tensor4_il_type = std::initializer_list<tensor3_il_type>;

    /** @brief Initializes *this by processing the input provided in @p input.
     *
     *  This ctor is only public to facilitate unit testing of the library.
     *  Users should ignore this ctor and focus on the variadic value ctor
     *  instead (which dispatches to this ctor).
     *
     *  @param[in] input An implementation-defined object containing the inputs
     *                   provided by the user.
     *
     *  @throw std::runtime_error if the inputs in @p input are not valid.
     *                            Strong throw guarantee.
     *
     *  @throw std::bad_alloc if there is a problem allocating the state for
     *                        *this. Strong throw guarantee.
     */
    explicit Tensor(input_type input);

    /** @brief Variadic value ctor.
     *
     *  @tparam Args The types of the arguments.
     *
     *  @note The intent is to create a tutorial showcasing how to initialize
     *        the Tensor object under different conditions and NOT for the user
     *        to have to reverse engineer the options. For now,
     *        detail_::TensorInput is the authority on what inputs are allowed
     *        and detail_::TensorFactory is the authority on what combinations
     *        and values are allowed.
     *
     *  This ctor is the default ctor (when @p Args is an empty parameter pack)
     *  and also the value ctor. Ultimately there are a lot of different ways
     *  to initialize a Tensor object. To simplify the API of the Tensor class
     *  we have defined a single ctor which works with all of them. Arguments
     *  to this ctor may be provided in any order and will be parsed by the
     *  backend. The backend is also responsible for determining appropriate
     *  defaults for the information given. While there are many valid
     *  input combinations, we anticipate users being most interested in the
     *  following use cases (for each use case the first phrase describes the
     *  inputs the ctor is invoked with and the remainder describes how *this
     *  will be initialized):
     *
     *  - No arguments. Creates an empty tensor. An empty tensor
     *    has no rank, and no elements. It is NOT a scalar. The resulting tensor
     *    primarily serves as a placeholder until it is initialized.
     *  - Logical layout. This use case includes providing a
     *    class from the layout::Logical family or the inputs necessary to build
     *    one (minimally a shape, but also optionally the symmetry and
     *    sparsity). This is the ctor we want users to eventually target. At
     *    present the mapping from the logical layout to the physical layout is
     *    naive and unlikely to result in good performance.
     *  - Logical and physical layouts. We'll call this expert initialization.
     *    At present this is the use case to target if you are concerned with
     *    performance. Given both the logical and physical layouts the backend
     *    will dispatch to the tensor library designed for your use case.
     *  - (NYI) A container of elements. Envisioned to be used primarily for
     *    testing, this use case allows you to provide the literal elements of
     *    the tensor and the backend will fill in the rest.
     *
     *  @param[in] args Zero or more inputs to use to initialize *this. A
     *                  compiler error will arise if a provided argument is not
     *                  a valid input type.
     *
     *  @throw std::runtime_error if at least one input is provided, but the
     *                            provided input(s) are insufficient to
     *                            initialize *this. Also raised if the provided
     *                            inputs are inconsistent. Strong throw
     *                            guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the state for
     *                        *this. Strong throw guarantee.
     */
    template<typename... Args>
    Tensor(disable_if_tensor_t<Args>&&... args) :
      Tensor(input_type(std::forward<Args>(args)...)) {}

    /** @brief Creates a tensor from a (possibly) nested initializer list.
     *
     *  By nesting initializer lists it is possible to specify the initial
     *  values for a tensor and the logical layout. For example providing a
     *  single floating-point value indicates that the tensor is a scalar.
     *  Providing an initializer list of floating-point values indicates the
     *  tensor is a  vector. Providing an initializer list of initializer lists
     *  of floating-point values creates a matrix, or in general @f$r@f$ nested
     *  initializer lists create a rank @f$r@f$ tensor.
     *
     *  @warning At present these methods do NOT support jagged tensors. It is
     *           possible to extend these methods to jagged tensors, but it is
     *           not yet implemented.
     *
     *  @note Because of how C++ resolves initializer lists we need to have the
     *        public API overloaded for every rank tensor we want to support or
     *        require the user to work out the il type themselves. All of these
     *        dispatch to the same backend.
     *
     *  @param[in] il A (possibly) nested initializer list containing the
     *                initial values for the tensor.
     *
     *  @throw std::runtime_error if @p il does not describe a smooth tensor.
     *                            Strong throw guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the return.
     *                        Strong throw guarantee.
     */
    ///@{
    explicit Tensor(scalar_il_type il);
    Tensor(vector_il_type il);
    Tensor(matrix_il_type il);
    Tensor(tensor3_il_type il);
    Tensor(tensor4_il_type il);
    ///@}

    /** @brief Initializes *this with a deep copy of @p other.
     *
     *  @param[in] other The tensor to copy.
     *
     *  @throw std::bad_alloc if there is a problem copying @p other. Strong
     *                        throw guarantee.
     */
    Tensor(const Tensor& other);

    /** @brief Initializes *this with the state in @p other.
     *
     *  @param[in,out] other The tensor to take the state from. After the call
     *                       @p other will be an empty tensor.
     *
     *  @throw None No throw guarantee.
     */
    Tensor(Tensor&& other) noexcept;

    /** @brief Overwrites the state of *this with a deep copy of @p rhs.
     *
     *  This method will release the state currently owned by *this and
     *  overwrite it with a deep copy of @p rhs. The copy will occur through
     *  Tensor's copy ctor, so see that method for more details.
     *
     *  @param[in] rhs The tensor to copy.
     *
     *  @return *this after replacing its state with a deep copy of @p rhs.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    Tensor& operator=(const Tensor& rhs);

    /** @brief Overwrites the state of *this with the state of @p rhs.
     *
     *  This method will release the state currently owned by *this and
     *  overwrite it with the state of @p rhs. The move will occur through
     *  Tensor's move ctor, so see that method for more details.
     *
     *  @param[in,out] rhs The tensor to take the state from. After this call
     *                     @p rhs will be an empty tensor.
     *
     *  @return *this after replacing its state with the state in @p rhs.
     *
     *  @throw None No throw guarantee.
     */
    Tensor& operator=(Tensor&& rhs) noexcept;

    /// Defaulted no-throw dtor
    ~Tensor() noexcept;

    /** @brief Read-only access to the tensor's logical layout.
     *
     *  The logical layout of a tensor is how the user is thinking about it.
     *  This is usually different from how the tensor is actually stored by the
     *  backend. When interacting with the Tensor class it is always done
     *  assuming the tensor has the logical layout (to interact with the tensor
     *  in the way it is actually laid out go through the buffer).
     *
     *  @return A read-only reference to the logical layout of the tensor.
     *
     *  @throw std::runtime_error if *this is an empty tensor. Strong throw
     *                            guarantee.
     */
    const_logical_reference logical_layout() const;

    /** @brief Read-only access to the tensor's buffer.
     *
     *  The buffer of a tensor contains the actual elements. Generally speaking,
     *  users should not have to interact with the buffer. The primary
     *  exception to this is if the user wants to interface TensorWrapper with
     *  another tensor solution.
     *
     *  @return A read-only reference to the buffer of the tensor.
     *
     *  @throw std::runtime_error if *this is an empty tensor. Strong throw
     *                            guarantee.
     */
    const_buffer_reference buffer() const;

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Exchanges the state of *this with the state of @p other.
     *
     *  @param[in,out] other The tensor to take the state from. After this call
     *                       @p other will contain the state which was
     *                       previously in *this.
     *
     *  @throw None No throw guarantee.
     */
    void swap(Tensor& other) noexcept;

    /** @brief Is *this value equal to @p rhs?
     *
     *  Two tensor objects are value equal if they are both empty tensors or if
     *  their respective logical layouts and buffers are polymorphically value
     *  equal. Of note this means that the floating point representation of the
     *  tensors' elements are compared for value equality and they must be
     *  exactly equal. It also means that even if two tensors have the same
     *  physical layout, if the user wants to think about them differently they
     *  will compare as not value equal.
     *
     *  @param[in] rhs The tensor to compare to.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const Tensor& rhs) const noexcept;

    /** @brief Is this different from @p rhs?
     *
     *  Two tensors are defined to be different if they are not value equal.
     *  See operator== for the definition of value==.
     *
     *  @param[in] rhs The tensor to compare to.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const Tensor& rhs) const noexcept;

private:
    /// All ctors ultimately dispatch to this ctor
    Tensor(pimpl_pointer pimpl) noexcept;

    /// Does *this have a PIMPL?
    bool has_pimpl_() const noexcept;

    /// Throws if *this does not have a PIMPL.
    void assert_pimpl_() const;

    /// Object actually implementing *this
    pimpl_pointer m_pimpl_;
};

} // namespace tensorwrapper
