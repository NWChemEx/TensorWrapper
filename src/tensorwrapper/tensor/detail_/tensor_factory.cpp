#include "tensor_pimpl.hpp"
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/tensor/detail_/tensor_factory.hpp>
namespace tensorwrapper::detail_ {

using pimpl_pointer = typename Tensor::pimpl_pointer;

using symmetry_pointer        = typename TensorFactory::symmetry_pointer;
using sparsity_pointer        = typename TensorFactory::sparsity_pointer;
using logical_layout_pointer  = typename TensorFactory::logical_layout_pointer;
using physical_layout_pointer = typename TensorFactory::physical_layout_pointer;
using allocator_pointer       = typename TensorFactory::allocator_pointer;
using buffer_pointer          = typename TensorFactory::buffer_pointer;

// -----------------------------------------------------------------------------
// -- Methods for determining reasonable defaults
// -----------------------------------------------------------------------------

symmetry_pointer TensorFactory::default_logical_symmetry(
  const_shape_reference) {
    // Symmetry is  at present NOT polymorphic
    return std::make_unique<symmetry_base>();
}

sparsity_pointer TensorFactory::default_logical_sparsity(
  const_shape_reference, const_symmetry_reference) {
    // Sparsity is  at present NOT polymorphic
    return std::make_unique<sparsity_base>();
}

logical_layout_pointer TensorFactory::default_logical_layout(
  shape_pointer pshape, symmetry_pointer psymmetry,
  sparsity_pointer psparsity) {
    using logical_type = input_type::logical_layout_type;

    return std::make_unique<logical_type>(
      std::move(pshape), std::move(psymmetry), std::move(psparsity));
}

physical_layout_pointer TensorFactory::default_physical_layout(
  const_logical_reference logical) {
    // For now the default physical layout is a copy of the logical layout
    return std::make_unique<physical_layout_type>(
      logical.shape(), logical.symmetry(), logical.sparsity());
}

allocator_pointer TensorFactory::default_allocator(
  const_physical_reference physical) {
    // For now, default allocator makes Eigen tensors filled with doubles
    const auto rank = physical.shape().rank();

    // N.B. all specializations implement make_eigen_allocator the same
    using eigen_alloc = allocator::Eigen<double, 0>;
    return eigen_alloc::make_eigen_allocator(rank, m_rv_);
}

// -----------------------------------------------------------------------------
// -- Construct
// -----------------------------------------------------------------------------

pimpl_pointer TensorFactory::construct(TensorInput input) {
    // N.B. Ultimately need a logical layout and a buffer. The former drives the
    // later so we make that first (if we don't have it).

    if(!input.has_logical_layout()) {
        if(!input.has_shape()) {
            // TODO: Could infer shape if given an initialization, but for now
            // treat it as default initialization
            return pimpl_pointer{};
        }
        const auto& shape = *input.m_pshape;

        if(!input.has_symmetry()) {
            input.m_psymmetry = default_logical_symmetry(shape);
        }

        if(!input.has_sparsity()) {
            const auto& symm  = *input.m_psymmetry;
            input.m_psparsity = default_logical_sparsity(shape, symm);
        }

        input.m_plogical = default_logical_layout(std::move(input.m_pshape),
                                                  std::move(input.m_psymmetry),
                                                  std::move(input.m_psparsity));
    }

    // We now have a logical layout. If we don't have a buffer we can make one
    // now.

    if(!input.has_buffer()) {
        if(!input.has_physical_layout()) {
            input.m_pphysical = default_physical_layout(*input.m_plogical);
        }

        if(!input.has_allocator()) {
            input.m_palloc = default_allocator(*input.m_pphysical);
        }

        // TODO: Check if we have initialization criteria
        input.m_pbuffer =
          input.m_palloc->allocate(std::move(input.m_pphysical));
    }

    // Now we have both a logical layout and a buffer so we're done

    return std::make_unique<pimpl_type>(std::move(input.m_plogical),
                                        std::move(input.m_pbuffer));
}

} // namespace tensorwrapper::detail_
