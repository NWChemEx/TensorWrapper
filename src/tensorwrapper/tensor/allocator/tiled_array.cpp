#include "tiled_array_allocator_helper.hpp"

#define TPARAM template<typename FieldType>
#define TA_ALLOCATOR TiledArrayAllocator<FieldType>
namespace tensorwrapper::tensor::allocator {

TPARAM void TA_ALLOCATOR::hash_(tensorwrapper::detail_::Hasher& h) const {
    throw std::runtime_error("TA Allocator Hash NYI");
}

TPARAM typename TA_ALLOCATOR::allocator_ptr TA_ALLOCATOR::clone_() const {
    return allocator_ptr(new my_type(*this));
}

TPARAM typename TA_ALLOCATOR::value_type TA_ALLOCATOR::allocate_(
  const tile_populator_type& fxn, const shape_type& shape) const {
    using default_tensor_type = detail_::default_tensor_type<FieldType>;
    default_tensor_type ta_tensor;
    if constexpr(field::is_scalar_field_v<FieldType>) {
        ta_tensor = detail_::generate_ta_scalar_tensor(this->m_world_, shape,
                                                       tiling_, fxn);
    } else {
        ta_tensor =
          detail_::generate_ta_tot_tensor(this->m_world_, shape, tiling_, fxn);
    }

    // Wrap in buffer PIMPL
    using ta_buffer_pimpl_type = detail_::ta_buffer_pimpl_type<FieldType>;
    ta_buffer_pimpl_type ta_buffer_pimpl(ta_tensor);

    // Return Buffer pointer
    return value_type(ta_buffer_pimpl.clone());
}

TPARAM bool TA_ALLOCATOR::is_equal_(const base_type& other) const noexcept {
    // Attempt to downcast
    auto ptr = dynamic_cast<const my_type*>(&other);

    // If this failed, it's not a TA Allocator
    if(!ptr) return false;

    return (*this) == (*ptr);
}

template class TiledArrayAllocator<field::Scalar>;
template class TiledArrayAllocator<field::Tensor>;
} // namespace tensorwrapper::tensor::allocator
