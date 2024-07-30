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
#include <parallelzone/parallelzone.hpp>
#include <tensorwrapper/buffer/buffer_base.hpp>
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/layout/physical.hpp>

namespace tensorwrapper::allocator {

/** @brief Common base class for all allocators.
 *
 *  The AllocatorBase class serves as type-erasure and a unified API for all
 *  allocators.
 */
class AllocatorBase : public detail_::PolymorphicBase<AllocatorBase> {
private:
    /// The type of *this
    using my_type = AllocatorBase;

    /// The type *this derives from
    using my_base_type = detail_::PolymorphicBase<my_type>;

public:
    /// Type of a view of the runtime system
    using runtime_view_type = parallelzone::runtime::RuntimeView;

    /// Type of a mutable reference to the runtime system
    using runtime_view_reference = runtime_view_type&;

    /// Type of a read-only reference to the runtime system
    using const_runtime_view_reference = const runtime_view_type&;

    /// Type all physical layouts derive from
    using layout_type = layout::Physical;

    /// Type of a pointer to an object of type layout_type
    using layout_pointer = typename layout_type::layout_pointer;

    /// Type all buffers derive from
    using buffer_base_type = buffer::BufferBase;

    /// Type of a pointer to an object of type buffer_base_type
    using buffer_base_pointer = typename buffer_base_type::buffer_base_pointer;

    // -------------------------------------------------------------------------
    // -- Ctors and assignment
    // -------------------------------------------------------------------------

    /** @brief Polymorphically allocates a new buffer.
     *
     *  This method type-erases the process of creating a buffer by dispatching
     *  to the derived class. In general the buffer created by this method will
     *  NOT be initialized, though this will depend on the default behavior of
     *  the backend. Use `construct` instead of `allocate` if you additionally
     *  want to guarantee initialization.
     *
     *  Derived classes implement this method by overriding allocate_.
     *
     *  @param[in] playout A pointer to the layout for the new buffer.
     *
     *  @return The newly allocated, but not necessarily initialized buffer.
     */
    buffer_base_pointer allocate(layout_pointer playout) {
        return allocate_(std::move(playout));
    }

    /** @brief The runtime *this uses for allocating.
     *
     *  Allocators are tied to runtimes. This method can be used to retrieve
     *  the runtime *this is using for allocation.
     *
     *  @return A mutable reference to the runtime *this is using for allocating
     *          buffers.
     *
     *  @throw None No throw guarantee.
     */
    runtime_view_reference runtime() noexcept { return m_rv_; }

    /** @brief The runtime *this uses for allocating.
     *
     *  This method is the same as the non-const version except that it returns
     *  the runtime in a read-only manner.
     *
     *  @return A read-only reference to the runtime *this uses for allocating
     *          buffers.
     *
     *  @throw None No throw guarantee.
     */
    const_runtime_view_reference runtime() const noexcept { return m_rv_; }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    /** @brief Is *this value equal to @p rhs?
     *
     *  This method is non-polymorphic and only compares the AllocatorBase part
     *  of *this to the AllocatorBase part of @p rhs. Two AllocatorBase objects
     *  are value equal if they contain views of the same runtime.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const AllocatorBase& rhs) const noexcept {
        return m_rv_ == rhs.m_rv_;
    }

    /** @brief Is *this different from @p rhs?
     *
     *  This method defines "different" as "not value equal." See the
     *  documentation for operator== for the definition of value equal.
     *
     *  @param[in] rhs The allocator to compare against.
     *
     *  @return False if *this is value equal to @p rhs and true otherwise.
     *
     *  @throw None No throw guarantee.
     *
     */
    bool operator!=(const AllocatorBase& rhs) const noexcept {
        return !((*this) == rhs);
    }

protected:
    /** @brief Creates an allocator for the runtime @p rv.
     *
     *  @param[in] rv The runtime in which to allocate buffers.
     *
     *  @throw None No throw guarantee.
     */
    explicit AllocatorBase(runtime_view_type rv) : m_rv_(std::move(rv)) {}
    /** @brief Creates *this so that it uses the same runtime as @p other.
     *
     *  @param[in] other The allocator to make a copy of.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    AllocatorBase(const AllocatorBase& other) = default;

    /** @brief Derived classes should overwrite in order to implement allocate.
     *
     *  Derived classes are charged with ensuring @p playout is a valid layout
     *  and then creating a buffer adhering to the layout.
     *
     *  @param[in] playout The layout for the buffer to allocate.
     *
     *  @throw std::bad_alloc if the allocation fails. Strong throw guarantee.
     *  @throw std::runtime_error if @p playout is not a valid layout. Strong
     *                            throw guarantee.
     */
    virtual buffer_base_pointer allocate_(layout_pointer playout) = 0;

private:
    /// The runtime we are allocating memory in
    runtime_view_type m_rv_;
};

} // namespace tensorwrapper::allocator
