#pragma once
#include <parallelzone/parallelzone.hpp>
#include <tensorwrapper/buffer/buffer_base.hpp>
#include <tensorwrapper/detail_/polymorphic_base.hpp>
#include <tensorwrapper/layout/tiled.hpp>

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

    /// Type all layouts derive from
    using layout_type = layout::Tiled;

    /// Type of a pointer to an object of type layout_type
    using layout_pointer = typename layout_type::layout_pointer;

    /// Type all buffers derive from
    using buffer_base_type = buffer::BufferBase;

    /// Type of a pointer to an object of type buffer_base_type
    using buffer_base_pointer = typename buffer_base_type::buffer_base_pointer;

    // -------------------------------------------------------------------------
    // -- Ctors and assignment
    // -------------------------------------------------------------------------

    explicit AllocatorBase(runtime_view_type rv) : m_rv_(std::move(rv)) {}

    buffer_base_pointer allocate(layout_pointer playout) {
        return allocate_(std::move(playout));
    }

    runtime_view_reference runtime() noexcept { return m_rv_; }

    const_runtime_view_reference runtime() const noexcept { return m_rv_; }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

    bool operator==(const AllocatorBase& rhs) const noexcept {
        return m_rv_ == rhs.m_rv_;
    }

    bool operator!=(const AllocatorBase& rhs) const noexcept {
        return !((*this) == rhs);
    }

protected:
    /** @brief Creates *this so that it uses the same runtime as @p other.
     *
     *  @param[in] other The allocator to make a copy of.
     *
     *  @throw std::bad_alloc if there is a problem allocating the copy. Strong
     *                        throw guarantee.
     */
    AllocatorBase(const AllocatorBase& other) = default;

    virtual buffer_base_pointer allocate_(layout_pointer playout) = 0;

private:
    /// The runtime we are allocating memory in
    runtime_view_type m_rv_;
};

} // namespace tensorwrapper::allocator
