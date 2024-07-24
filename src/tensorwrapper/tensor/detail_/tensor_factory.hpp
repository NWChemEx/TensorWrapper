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
#include <memory>
#include <tensorwrapper/tensor/tensor_class.hpp>
namespace tensorwrapper::detail_ {

/** @brief Object which helps create tensor objects.
 *
 *  Ultimately there are going to be a lot of possible ways to create a tensor.
 *  In an effort to keep the Tensor's ctors as simple as possible we have
 *  decoupled the set of all possible inputs into the TensorInput class and the
 *  set of all possible initialization methods into the TensorFactory class.
 */
class TensorFactory {
public:
    /// Type this will create PIMPLs for
    using tensor_type = Tensor;

    /// Type used to manage the myriad of possible inputs
    using input_type = TensorInput;

    /// Type powering an object of type tensor_type
    using pimpl_type = typename tensor_type::pimpl_type;

    /// Type of a pointer to an object of type pimpl_type
    using pimpl_pointer = typename tensor_type::pimpl_pointer;

    // Pull types from input_type that we will need for our API
    using const_shape_reference    = input_type::const_shape_reference;
    using shape_pointer            = input_type::shape_pointer;
    using const_symmetry_reference = input_type::const_symmetry_reference;
    using symmetry_pointer         = input_type::symmetry_pointer;
    using const_sparsity_reference = input_type::const_sparsity_reference;
    using sparsity_pointer         = input_type::sparsity_pointer;
    using const_logical_reference  = input_type::const_logical_reference;
    using logical_layout_pointer   = input_type::logical_layout_pointer;
    using const_physical_reference = input_type::const_physical_reference;
    using physical_layout_pointer  = input_type::physical_layout_pointer;
    using allocator_pointer        = input_type::allocator_pointer;
    using runtime_view_type        = input_type::runtime_view_type;

    // -------------------------------------------------------------------------
    // -- Methods for determining reasonable defaults
    // -------------------------------------------------------------------------

    /** @brief Determines a default logical symmetry for @p shape.
     *
     *  Given @p shape this method is charged with determining a corresponding
     *  default symmetry. Without access to the elements there is no way *this
     *  can actually determine the full symmetry and thus this method always
     *  returns a default constructed symmetry group, i.e., the resulting tensor
     *  will have no symmetry.
     *
     *  @param[in] shape The shape to compute the symmetry of.
     *
     *  @return The symmetry group for a tensor with no symmetry.
     *
     *  @throw std::bad_alloc if there is a problem allocating the new group.
     *                        Strong throw guarantee.
     */
    static symmetry_pointer default_logical_symmetry(
      const_shape_reference shape);

    /** @brief Constructs a default sparsity from the shape and symmetry.
     *
     *  At present sparsity is a stub class and this function simply returns a
     *  default constructed instance.
     *
     *  @param[in] shape The tensor's logical shape.
     *  @param[in] symmetry The tensor's logical symmetry.
     *
     *  @return The tensor's logical sparsity.
     *
     *  @throw std::bad_alloc if there is a problem allocating the new group.
     *                        Strong throw guarantee.
     */
    static sparsity_pointer default_logical_sparsity(
      const_shape_reference shape, const_symmetry_reference symmetry);

    /** @brief Constructs the tensor's default logical layout.
     *
     *  Logical layouts are simply wrappers around the shape, symmetry, and
     *  sparsity. This method simply forwards the inputs to the new layout
     *  object.
     *
     *  @param[in] shape The tensor's logical shape.
     *  @param[in] symmetry The tensor's logical symmetry.
     *  @param[in] sparsity The tensor's logical sparsity.
     *
     *  @return The tensor's logical layout.
     *
     *  @throw std::bad_alloc if there is a problem allocating the return.
     */
    static logical_layout_pointer default_logical_layout(
      shape_pointer shape, symmetry_pointer symmetry,
      sparsity_pointer sparsity);

    /** @brief Construct's the tensor's default physical layout.
     *
     *  At present the default physical layout for a tensor is the same as its
     *  logical layout. Eventually this should take runtime conditions into
     *  account.
     *
     *  @param[in] logical The logical layout of the tensor.
     *
     *  @return The default physical layout for the tensor.
     *
     *  @throw std::bad_alloc if there is a problem allocating the return.
     */
    static physical_layout_pointer default_physical_layout(
      const_logical_reference logical);

    /**  @brief Constructs an allocator consistent with the physical layout.
     *
     *   @param[in] physical The physical layout of the tensor we want to
     *                       allocate.
     *   @param[in] rv The runtime that tensors will be allocated in.
     *
     *   @return An allocator capable of allocating a tensor with the layout
     *           @p physical using the resources in @p rv.
     *
     *   @throw std::bad_alloc if there is a problem allocating the return.
     *                         Strong throw guarantee.
     */
    static allocator_pointer default_allocator(
      const_physical_reference physical, runtime_view_type rv);

    /** @brief Actually constructs the tensor's PIMPL.
     *
     *  This is the main entry point into this class (and is what callers
     *  should use). Since this class is not user-facing we have opted to have
     *  all methods be public (to assist with unit testing).
     *
     *  @param[in] input The objects the user wants us to use to construct the
     *                   tensor.
     *
     *  @throw std::runtime_error if @p input is not in a valid state. Strong
     *                            throw guarantee.
     *  @throw std::bad_alloc if there is a problem allocating the return.
     *                            Strong throw guarantee.
     */
    static pimpl_pointer construct(input_type input);

    // -------------------------------------------------------------------------
    // -- Assessing input validity
    // -------------------------------------------------------------------------

    /** @brief Does @p input contain sufficient information in order for *this
     *         to make a logical layout?
     *
     *  At present *this can make a logical layout given at least a shape. It
     *  also can trivially make a logical layout if it is given a logical
     *  layout.
     *
     *  @param[in] input The inputs we are inspecting.
     *
     *  @return True if @p input contains sufficient information to create a
     *          logical layout and false otherwise.
     */
    static bool can_make_logical_layout(const input_type& input) noexcept;

    /** @brief Throws if @p input has been constructed in an invalid state.
     *
     *  Assuming @p input manages @f$N@f$ inputs there are @f$2^N@f$ possible
     *  states for @p input just considering whether each pointer is null or
     *  not. While *this is capable of working out defaults for some inputs,
     *  this is not true for all inputs. For example, there is no
     *  way for *this to work out the logical shape of a tensor given only its
     *  logical symmetry. The goal of this method is to wrap the process of
     *  determining whether or not `construct` has a chance of succeeding before
     *  we start constructing.
     *
     *  @param[in] input The input we are error-checking.
     *
     *  @throw std::runtime_error if @p input has been constructed in an invalid
     *                            state. Strong throw guarantee.
     */
    static void assert_valid(const input_type& input);
};

} // namespace tensorwrapper::detail_
