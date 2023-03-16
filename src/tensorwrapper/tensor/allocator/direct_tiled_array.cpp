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

#define TPARAM template<typename FieldType>
#define TA_ALLOCATOR DirectTiledArrayAllocator<FieldType>
namespace tensorwrapper::tensor::allocator {

TPARAM typename TA_ALLOCATOR::allocator_ptr TA_ALLOCATOR::clone_() const {
    return allocator_ptr(new my_type(*this));
}

TPARAM typename TA_ALLOCATOR::value_pointer TA_ALLOCATOR::allocate_(
  const tile_populator_type& fxn, const shape_type& shape) const {
    using lazy_tensor_type     = detail_::lazy_tensor_type<FieldType>;
    using ta_buffer_pimpl_type = detail_::ta_buffer_pimpl_type<FieldType>;

    runtime_type m_world_{};
    lazy_tensor_type ta_tensor;
    if constexpr(field::is_scalar_field_v<FieldType>) {
        ta_tensor = detail_::generate_ta_scalar_direct_tensor(
          m_world_.madness_world(), shape, m_fxn_id_, fxn);
    } else {
        ta_tensor = detail_::generate_ta_tot_direct_tensor(
          m_world_.madness_world(), shape, m_fxn_id_, fxn);
    }

    // Return Buffer pointer
    return std::make_unique<value_type>(
      std::make_unique<ta_buffer_pimpl_type>(ta_tensor));
}

TPARAM typename TA_ALLOCATOR::value_pointer TA_ALLOCATOR::allocate_(
  const element_populator_type& fxn, const shape_type& shape) const {
    using lazy_tensor_type     = detail_::lazy_tensor_type<FieldType>;
    using ta_buffer_pimpl_type = detail_::ta_buffer_pimpl_type<FieldType>;

    runtime_type m_world_{};
    lazy_tensor_type ta_tensor;
    if constexpr(field::is_scalar_field_v<FieldType>) {
        ta_tensor = detail_::generate_ta_scalar_direct_tensor(
          m_world_.madness_world(), shape, m_fxn_id_, fxn);
    } else {
        ta_tensor = detail_::generate_ta_tot_direct_tensor(
          m_world_.madness_world(), shape, m_fxn_id_, fxn);
    }

    // Return Buffer pointer
    return std::make_unique<value_type>(
      std::make_unique<ta_buffer_pimpl_type>(ta_tensor));
}

TPARAM typename TA_ALLOCATOR::value_pointer TA_ALLOCATOR::reallocate_(
  const value_type& buf, const shape_type& shape) const {
    /** Can't retile a lazy array and can't turn a data array into a
     *  lazy array, so this just returns a buffer with a lazy array that
     *  has the provided shape and tiles that evaluate using m_fxn_id_.
     *  Potential gotcha here: Assumes that m_fxn_id_ was previously
     *  registered with the LazyTile map of functions (i.e. through an
     *  allocate call).
     */
    tile_populator_type fxn;
    return allocate_(fxn, shape);
}

TPARAM bool TA_ALLOCATOR::is_equal_(const base_type& other) const noexcept {
    // Attempt to downcast
    auto ptr = dynamic_cast<const my_type*>(&other);

    // If this failed, it's not a TA Allocator
    if(!ptr) return false;

    return (*this) == (*ptr);
}

template class DirectTiledArrayAllocator<field::Scalar>;
template class DirectTiledArrayAllocator<field::Tensor>;
} // namespace tensorwrapper::tensor::allocator
