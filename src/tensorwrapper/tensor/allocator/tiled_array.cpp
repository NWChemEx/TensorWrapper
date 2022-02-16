#include "tensorwrapper/tensor/allocators/tiled_array.hpp"
#include "../../sparse_map/sparse_map/detail_/tiling_map_index.hpp"
#include "tiled_array_tiling.hpp"
#include "tiled_array_sparse_shape.hpp"

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
  const scalar_populator_type& fxn, const shape_type& shape ) const {

    // Get TiledRange for the specified tiling
    auto ta_trange = detail_::make_tiled_range(tiling_, shape);

    // Get the TA Shape
    //auto ta_shape = detail_::make_sparse_shape<FieldType>(shape, ta_trange);
#if 1

    // Create TA tensor
    using default_tensor_type = detail_::default_tensor_type<FieldType>;
    default_tensor_type ta_tensor;
    if constexpr (std::is_same_v<FieldType,field::Scalar>) {
      if( fxn ) {
          auto ta_functor = [&](TA::Tensor<double>& t, TA::Range const& range) {
              t = TA::Tensor<double>(range, 0.0);
              fxn( range.lobound(), range.upbound(), t.data());
              return TA::norm(t);
          };
          ta_tensor = TA::make_array<default_tensor_type>(this->m_world_,
            ta_trange, ta_functor);
      } else {
          //ta_tensor = default_tensor_type(this->m_world_, ta_trange, ta_shape );
      }
    } else {
      throw std::runtime_error("Haven't worked out ToT population yet...");
    }

    // Wrap in buffer PIMPL
    using ta_buffer_pimpl_type = detail_::ta_buffer_pimpl_type<FieldType>;
    ta_buffer_pimpl_type ta_buffer_pimpl( ta_tensor );

    // Return Buffer pointer
    return value_type( ta_buffer_pimpl.clone() );
#else
    return value_type();
#endif
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
}
