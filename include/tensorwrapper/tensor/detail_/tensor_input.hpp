#pragma once
#include <memory>
#include <parallelzone/parallelzone.hpp>
#include <tensorwrapper/allocator/allocator_base.hpp>
#include <tensorwrapper/layout/layout_base.hpp>
#include <tensorwrapper/layout/logical.hpp>
#include <tensorwrapper/layout/physical.hpp>

namespace tensorwrapper::detail_ {

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

    /// Type of a pointer to an object of type allocator_base
    using allocator_pointer = typename allocator_base::base_pointer;

    /// Type all buffer object's inherit from
    using buffer_base = typename allocator_base::buffer_base_type;

    /// Type of a read-only reference to an object of type buffer_base
    using const_buffer_reference = typename buffer_base::const_base_reference;

    /// Type of a pointer to an object of type buffer_base
    using buffer_pointer = typename buffer_base::base_pointer;

    /// Type of a view of the runtime
    using runtime_view_type = typename allocator_base::runtime_view_type;

    TensorInput() = default;

    template<typename... Args>
    TensorInput(const_shape_reference shape, Args&&... args) :
      TensorInput(shape.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(shape_pointer pshape, Args&&... args) :
      TensorInput(std::forward<Args>(args)...), m_pshape(std::move(pshape)) {}

    template<typename... Args>
    TensorInput(const_symmetry_reference symmetry, Args&&... args) :
      TensorInput(std::make_unique<symmetry_base>(),
                  std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(symmetry_pointer psymmetry, Args&&... args) :
      TensorInput(std::forward<Args>(args)...),
      m_psparsity(std::move(psymmetry)) {}

    template<typename... Args>
    TensorInput(const_sparsity_reference sparsity, Args&&... args) :
      TensorInput(std::make_unique<sparsity_base>(),
                  std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(sparsity_pointer psparsity, Args&&... args) :
      TensorInput(std::forward<Args>(args)...),
      m_psparsity(std::move(psparsity)) {}

    template<typename... Args>
    TensorInput(const_logical_reference logical, Args&&... args) :
      TensorInput(logical.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(const_logical_pointer plogical, Args&&... args) :
      TensorInput(std::forward<Args>(args)...),
      m_plogical(std::move(plogical)) {}

    template<typename... Args>
    TensorInput(const_physical_reference physical, Args&&... args) :
      TensorInput(physical.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(physical_pointer pphysical, Args&&... args) :
      TensorInput(std::forward<Args>(args)...),
      m_pphysical(std::move(pphysical)) {}

    template<typename... Args>
    TensorInput(const_allocator_reference alloc, Args&&... args) :
      TensorInput(alloc.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(allocator_pointer palloc, Args&&... args) :
      TensorInput(std::forward<Args>(args)...), m_palloc(std::move(palloc)) {}

    template<typename... Args>
    TensorInput(const_buffer_reference buffer, Args&&... args) :
      TensorInput(buffer.clone(), std::forward<Args>(args)...) {}

    template<typename... Args>
    TensorInput(buffer_pointer pbuffer, Args&&... args) :
      m_pbuffer(std::move(pbuffer)), TensorInput(std::forward<Args>(args)...) {}

    template<typanem... Args>
    TensorInput(runtime_view_type rv, Args&&... args) :
      m_rv(std::move(rv)), TensorInput(std::forward<Args>(args)...) {}

    bool has_shape() const noexcept { return m_pshape != nullptr; }

    bool has_symmetry() const noexcept { return m_psymmetry != nullptr; }

    bool has_sparsity() const noexcept { return m_psparsity != nullptr; }

    bool has_logical_layout() const noexcept { return m_plogical != nullptr; }

    bool has_physical_layout() const noexcept { return m_pphysical != nullptr; }

    bool has_allocator() const noexcept { return m_palloc != nullptr; }

    bool has_buffer() const noexcept { return m_pbuffer != nullptr; }

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
