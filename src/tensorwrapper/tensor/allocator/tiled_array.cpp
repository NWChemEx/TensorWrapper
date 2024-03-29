/*
 * Copyright 2022 NWChemEx-Project
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

#include "tiled_array_allocator_helper.hpp"
#include <madness/world/MADworld.h>

#define TPARAM template<typename FieldType>
#define TA_ALLOCATOR TiledArrayAllocator<FieldType>
namespace tensorwrapper::tensor::allocator {

TPARAM typename TA_ALLOCATOR::allocator_ptr TA_ALLOCATOR::clone_() const {
    return allocator_ptr(new my_type(*this));
}

TPARAM typename TA_ALLOCATOR::value_pointer TA_ALLOCATOR::allocate_(
  const tile_populator_type& fxn, const shape_type& shape) const {
    using default_tensor_type  = detail_::default_tensor_type<FieldType>;
    using ta_buffer_pimpl_type = detail_::ta_buffer_pimpl_type<FieldType>;

    runtime_type m_runtime_{};
    auto comm   = m_runtime_.mpi_comm();
    auto& world = *madness::World::find_instance(SafeMPI::Intracomm(comm));
    default_tensor_type ta_tensor;
    if constexpr(field::is_scalar_field_v<FieldType>) {
        ta_tensor = detail_::generate_ta_scalar_tensor(world, shape, fxn);
    } else {
        ta_tensor = detail_::generate_ta_tot_tensor(world, shape, fxn);
    }

    // Return Buffer pointer
    return std::make_unique<value_type>(
      std::make_unique<ta_buffer_pimpl_type>(ta_tensor));
}

TPARAM typename TA_ALLOCATOR::value_pointer TA_ALLOCATOR::allocate_(
  const element_populator_type& fxn, const shape_type& shape) const {
    using default_tensor_type  = detail_::default_tensor_type<FieldType>;
    using ta_buffer_pimpl_type = detail_::ta_buffer_pimpl_type<FieldType>;

    runtime_type m_runtime_{};
    auto comm   = m_runtime_.mpi_comm();
    auto& world = *madness::World::find_instance(SafeMPI::Intracomm(comm));
    default_tensor_type ta_tensor;
    if constexpr(field::is_scalar_field_v<FieldType>) {
        ta_tensor = detail_::generate_ta_scalar_tensor(world, shape, fxn);
    } else {
        ta_tensor = detail_::generate_ta_tot_tensor(world, shape, fxn);
    }

    // Return Buffer pointer
    return std::make_unique<value_type>(
      std::make_unique<ta_buffer_pimpl_type>(ta_tensor));
}

TPARAM typename TA_ALLOCATOR::value_pointer TA_ALLOCATOR::reallocate_(
  const value_type& buf, const shape_type& shape) const {
    using default_tensor_type  = detail_::default_tensor_type<FieldType>;
    using ta_buffer_pimpl_type = detail_::ta_buffer_pimpl_type<FieldType>;

    // TODO: Revisit after #40 is resolved
    auto buf_copy_uptr = buf.pimpl()->clone();
    auto buf_copy_ptr =
      dynamic_cast<ta_buffer_pimpl_type*>(buf_copy_uptr.get());
    if(!buf_copy_ptr)
        throw std::runtime_error(
          "TA Allocator:: Passed Buffer is not TA buffer");

    // Get new TR, etc
    auto ta_range = detail_::make_tiled_range(shape);
    // TODO Handle possible sparse_map driven shape

    // Retile
    buf_copy_ptr->retile(ta_range);

    // Create a new buffer
    return std::make_unique<value_type>(std::move(buf_copy_uptr));
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
