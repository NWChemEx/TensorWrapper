#pragma once
#include "../buffer/detail_/ta_buffer_pimpl.hpp"
#include <tensorwrapper/tensor/allocators/allocators.hpp>
#include <tensorwrapper/tensor/detail_/pimpl.hpp>
#include <tensorwrapper/tensor/fields.hpp>
#include <tensorwrapper/tensor/shapes/shape.hpp>

namespace tensorwrapper::tensor::detail_ {

template<typename T, typename FieldType = field::Scalar>
auto ta_to_tw(T&& t) {
    static_assert(field::is_scalar_field_v<FieldType>,
                  "Only scalar fields are presently implemented");

    if(!t.is_initialized()) return TensorWrapper<FieldType>();

    // Step 0: Make shape
    using shape_type   = Shape<FieldType>;
    using extents_type = typename shape_type::extents_type;
    extents_type extents(t.trange().rank(), 0);
    for(std::size_t i = 0; i < extents.size(); ++i)
        extents[i] = t.trange().elements_range().extent(i);
    auto pshape = std::make_unique<shape_type>(extents);

    // Step 1: Make allocator
    auto palloc = allocator::ta_allocator<FieldType>();

    // Step 2: TA tensor into a TABufferPIMPL
    using ta_pimpl_type = buffer::detail_::TABufferPIMPL<FieldType>;
    auto pt             = std::make_unique<ta_pimpl_type>(std::forward<T>(t));

    // Step 3: Move BufferPIMPL into a Buffer
    using buffer_type = buffer::Buffer<FieldType>;
    auto pbuffer      = std::make_unique<buffer_type>(std::move(pt));

    // Step 4: Move buffer, shape, and allocator into TensorWrapperPIMPL
    using pimpl_type = TensorWrapperPIMPL<FieldType>;
    auto ppimpl      = std::make_unique<pimpl_type>(
      std::move(pbuffer), std::move(pshape), std::move(palloc));

    // Finally make the tensor
    return TensorWrapper<FieldType>(std::move(ppimpl));
}

} // namespace tensorwrapper::tensor::detail_
