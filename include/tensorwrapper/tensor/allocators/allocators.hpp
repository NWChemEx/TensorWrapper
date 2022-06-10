#pragma once
#include "tensorwrapper/tensor/allocators/allocator.hpp"
#include "tensorwrapper/tensor/allocators/tiled_array.hpp"

namespace tensorwrapper::tensor : {

    namespace allocator {

    template<typename FieldType, typename... Args>
    typename Allocator<FieldType>::allocator_ptr ta_allocator(Args&&... args) {
        using alloc_type = TiledArrayAllocator<FieldType>;
        return std::make_unique<alloc_type>(std::forward<Args>(args)...);
    }

    } // namespace allocator

    /** @brief Wraps the process of creating a defaulted allocator
     *
     *  As more advanced alloctors are added to the library it will make sense
     * to change the default TensorWrapper allocator. To decouple code from this
     *  choice, classes and functions needing a default_allocator instance are
     *  encouraged to get that allocator from this function.
     *
     *  @tparam FieldType The type of the field the tensor is over. Assumed to
     * be either field::Scalar or field::Tensor.
     *
     *  @return a type-erased, allocator.
     *
     *  @throw std::bad_alloc if allocation fails. Strong throw guarantee.
     */
    template<typename FieldType>
    typename allocator::Allocator<FieldType>::allocator_ptr
    default_allocator() {
        //    return
        //    std::make_unique<allocator::TiledArrayAllocator<FieldType>>();
        return allocator::ta_allocator<FieldType>();
    }

} // namespace tensorwrapper::tensor:
