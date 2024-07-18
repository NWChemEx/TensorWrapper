#include "tensor_pimpl.hpp"
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/tensor/detail_/tensor_factory.hpp>
namespace tensorwrapper::detail_ {

using pimpl_pointer           = typename TensorFactory::pimpl_pointer;
using shape_pointer           = typename TensorFactory::shape_pointer;
using symmetry_pointer        = typename TensorFactory::symmetry_pointer;
using sparsity_pointer        = typename TensorFactory::sparsity_pointer;
using logical_layout_pointer  = typename TensorFactory::logical_layout_pointer;
using physical_layout_pointer = typename TensorFactory::physical_layout_pointer;
using buffer_pointer          = typename TensorFactory::buffer_pointer;

pimpl_pointer TensorFactory::construct(logical_layout_pointer plogical) {
    // Check if this is a default construction, if so, early out
    if(plogical == nullptr) return pimpl_pointer();

    // For now the default physical layout is a copy of the logical layout
    auto pphysical = std::make_unique<physical_layout_type>(
      plogical->shape(), plogical->symmetry(), plogical->sparsity());

    return construct(std::move(plogical), std::move(pphysical));
}

pimpl_pointer TensorFactory::construct(logical_layout_pointer plogical,
                                       physical_layout_pointer pphysical) {
    // Check if this is a default construction, if so, early out
    if(plogical == nullptr) return pimpl_pointer();

    // Verify physical layout == logical (required for now)
    using const_layout_base         = const layout::LayoutBase&;
    const_layout_base logical_base  = *plogical;
    const_layout_base physical_base = *pphysical;

    if(logical_base != physical_base)
        throw std::runtime_error(
          "For now logical and physical layouts can not differ.");

    // Default allocator makes Eigen tensors filled with doubles
    // N.B. all specializations implement make_eigen_allocator the same
    using eigen_alloc = allocator::Eigen<double, 0>;

    auto rank   = pphysical->shape().rank();
    auto palloc = eigen_alloc::make_eigen_allocator(rank, m_rv_);

    auto pbuffer = palloc->allocate(std::move(pphysical));

    return construct(std::move(plogical), std::move(pbuffer));
}

pimpl_pointer TensorFactory::construct(logical_layout_pointer plogical,
                                       buffer_pointer pbuffer) {
    // Check if this is a default construction, if so, early out
    if(plogical == nullptr) return pimpl_pointer();

    return std::make_unique<pimpl_type>(std::move(plogical),
                                        std::move(pbuffer));
}

} // namespace tensorwrapper::detail_
