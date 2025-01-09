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
#include <parallelzone/parallelzone.hpp>
#include <tensorwrapper/allocator/allocator_base.hpp>
#include <tensorwrapper/layout/layout_base.hpp>
#include <tensorwrapper/layout/logical.hpp>
#include <tensorwrapper/layout/physical.hpp>

namespace tensorwrapper::detail_ {

/** @brief Type capable of holding all valid inputs to a Tensor's ctor.
 *
 *  There are a lot of different ways to construct a Tensor. To decouple the
 *  construction logic from Tensor class we first introduce the TensorInput
 *  class. Conceptually this class is a std::tuple with one slot per valid
 *  input. We have added some small convenience functions on top of the tuple,
 *  but otherwise this class primarily exists to perform the template meta-
 *  programming necessary to get the input into a consistent order. Validity
 *  checks are not the responsibility of *this, but rather the TensorFactory
 *  class (as it is the TensorFactory class which knows what it can compute
 *  defaults for).
 *
 *  @note This class is an implementation detail and should NOT be created
 *        directly by the user.
 */
struct TensorInput {
    /// Type common to all layouts (used to get types of shape, sparsity, etc.)
    using layout_base = layout::LayoutBase;

    /// Type all shapes inherit from
    using shape_base = typename layout_base::shape_base;

    /// Type of a read-only reference to an object of type shape_base
    using const_shape_reference = typename layout_base::const_shape_reference;

    /// Type of a pointer to an object of type shape_base
    using shape_pointer = typename shape_base::base_pointer;

    /// Type of a symmetry object
    using symmetry_base = typename layout_base::symmetry_type;

    /// Type of a read-only reference to an object of type symmetry_base
    using const_symmetry_reference =
      typename layout_base::const_symmetry_reference;

    /// Type of a pointer to an object of type symmetry_type
    using symmetry_pointer = std::unique_ptr<symmetry_base>;

    /// Type all sparsity patterns inherit from
    using sparsity_base = typename layout_base::sparsity_type;

    /// Type of a read-only reference to an object of type sparsity_base
    using const_sparsity_reference = const sparsity_base&;

    /// Type of a pointer to an object of type sparsity_base
    using sparsity_pointer = std::unique_ptr<sparsity_base>;

    /// Type all logical layouts inherit from
    using logical_layout_type = layout::Logical;

    /// Type of a read-only reference to a logical_layout_type
    using const_logical_reference = const logical_layout_type;

    /// Type of a pointer to an object of type logical_layout_type
    using logical_layout_pointer = std::unique_ptr<logical_layout_type>;

    /// Type all physical layouts inherit from
    using physical_layout_type = layout::Physical;

    /// Type of a read-only reference to an object of type physical_layout_type
    using const_physical_reference = const physical_layout_type&;

    /// Type of a pointer to an object of type physical_layout_type
    using physical_layout_pointer = std::unique_ptr<physical_layout_type>;

    /// Type all allocators inherit from
    using allocator_base = allocator::AllocatorBase;

    /// Type of a read-only reference to an object of type allocator_base
    using const_allocator_reference =
      typename allocator_base::const_base_reference;

    /// Type of a pointer to an object of type allocator_base
    using allocator_pointer = typename allocator_base::base_pointer;

    /// Type all buffer object's inherit from
    using buffer_base = typename allocator_base::buffer_base_type;

    /// Type of a mutable reference to a buffer_base object
    using buffer_reference = typename buffer_base::base_reference;

    /// Type of a read-only reference to an object of type buffer_base
    using const_buffer_reference = typename buffer_base::const_base_reference;

    /// Type of a pointer to an object of type buffer_base
    using buffer_pointer = typename buffer_base::base_pointer;

    /// Type of a pointer to a read-only buffer_base object
    using const_buffer_pointer = typename buffer_base::const_base_pointer;

    /// Type of a view of the runtime
    using runtime_view_type = typename allocator_base::runtime_view_type;

    TensorInput() = default;

    /** @brief Recursively unpacks the variadic arguments given to the ctor.
     *
     *  @tparam Args The types of the arguments which still need to be
     *               processed. Each type in @p Args must be a type that can be
     *               handled by one of the overloads in this section.
     *
     *  Ctors in this section are selected when the user passes `n>0` arguments.
     *  The overload matching the type of the 0-th argument is selected and
     *  then the remaining `n-1` arguments are forwarded and the process
     *  repeats. Once 0 arguments are forwarded, the default ctor is selected
     *  and the recursion process unwinds. During the unwinding each overload
     *  maps its input to the appropriate member object.
     *
     *  @param[in] <name varies> The argument to process by this overload.
     *  @param[in] args The arguments which still need to be processed.
     *
     *  @throw ??? In general throws will occur if an allocation fails. Strong
     *             throw guarantee.
     */
    //@{
    template<typename... Args>
    TensorInput(const_shape_reference shape, Args&&... args) :
      TensorInput(shape.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(shape_pointer pshape, Args&&... args) :
      TensorInput(std::forward<Args>(args)...) {
        m_pshape = std::move(pshape);
    }

    template<typename... Args>
    TensorInput(const_symmetry_reference symmetry, Args&&... args) :
      TensorInput(symmetry.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(symmetry_pointer psymmetry, Args&&... args) :
      TensorInput(std::forward<Args>(args)...) {
        m_psymmetry = std::move(psymmetry);
    }

    template<typename... Args>
    TensorInput(const_sparsity_reference sparsity, Args&&... args) :
      TensorInput(sparsity.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(sparsity_pointer psparsity, Args&&... args) :
      TensorInput(std::forward<Args>(args)...) {
        m_psparsity = std::move(psparsity);
    }

    template<typename... Args>
    TensorInput(const_logical_reference logical, Args&&... args) :
      TensorInput(logical.clone_as<logical_layout_type>(),
                  std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(logical_layout_pointer plogical, Args&&... args) :
      TensorInput(std::forward<Args>(args)...) {
        m_plogical = std::move(plogical);
    }

    template<typename... Args>
    TensorInput(const_physical_reference physical, Args&&... args) :
      TensorInput(physical.clone_as<physical_layout_type>(),
                  std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(physical_layout_pointer pphysical, Args&&... args) :
      TensorInput(std::forward<Args>(args)...) {
        m_pphysical = std::move(pphysical);
    }

    template<typename... Args>
    TensorInput(const_allocator_reference alloc, Args&&... args) :
      TensorInput(alloc.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(allocator_pointer palloc, Args&&... args) :
      TensorInput(std::forward<Args>(args)...) {
        m_palloc = std::move(palloc);
    }

    template<typename... Args>
    TensorInput(const_buffer_reference buffer, Args&&... args) :
      TensorInput(buffer.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(buffer_pointer pbuffer, Args&&... args) :
      TensorInput(std::forward<Args>(args)...) {
        m_pbuffer = std::move(pbuffer);
    }

    template<typename... Args>
    TensorInput(runtime_view_type rv, Args&&... args) :
      TensorInput(std::forward<Args>(args)...) {
        m_rv = std::move(rv);
    }
    ///@}

    /** @brief Does *this have non-null pointers for a particular property?
     *
     *  The methods in this section are convenience methods for determining if
     *  a property of *this has been set. For example, `has_shape` checks if
     *  `m_pshape` is non-null.
     *
     *  @return True if the namesake property's corresponding pointer is
     *          non-null and false otherwise.
     *
     *  @throw None No throw guarantee.
     *
     */
    ///@{
    bool has_shape() const noexcept { return m_pshape != nullptr; }

    bool has_symmetry() const noexcept { return m_psymmetry != nullptr; }

    bool has_sparsity() const noexcept { return m_psparsity != nullptr; }

    bool has_logical_layout() const noexcept { return m_plogical != nullptr; }

    bool has_physical_layout() const noexcept { return m_pphysical != nullptr; }

    bool has_allocator() const noexcept { return m_palloc != nullptr; }

    bool has_buffer() const noexcept { return m_pbuffer != nullptr; }
    ///@}

    shape_pointer m_pshape;

    symmetry_pointer m_psymmetry;

    sparsity_pointer m_psparsity;

    logical_layout_pointer m_plogical;

    physical_layout_pointer m_pphysical;

    allocator_pointer m_palloc;

    buffer_pointer m_pbuffer;

    runtime_view_type m_rv;
};

} // namespace tensorwrapper::detail_
