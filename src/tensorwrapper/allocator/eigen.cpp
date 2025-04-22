/*
 * Copyright 2024 NWChemEx-Project
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

#include "../buffer/detail_/eigen_tensor.hpp"
#include "../tensor/detail_/il_utils.hpp"
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/detail_/unique_ptr_utilities.hpp>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::allocator {

#define TPARAMS template<typename FloatType>
#define EIGEN Eigen<FloatType>

TPARAMS
bool EIGEN::can_rebind(const_buffer_base_reference buffer) {
    auto pbuffer = dynamic_cast<const buffer::Eigen<FloatType>*>(&buffer);
    return pbuffer != nullptr;
}

TPARAMS
typename EIGEN::eigen_buffer_reference EIGEN::rebind(
  buffer_base_reference buffer) {
    if(can_rebind(buffer)) return static_cast<eigen_buffer_reference>(buffer);
    throw std::runtime_error("Can not rebind buffer");
}

TPARAMS
typename EIGEN::const_eigen_buffer_reference EIGEN::rebind(
  const_buffer_base_reference buffer) {
    if(can_rebind(buffer))
        return dynamic_cast<const_eigen_buffer_reference>(buffer);
    throw std::runtime_error("Can not rebind buffer");
}

// -----------------------------------------------------------------------------
// -- Protected methods
// -----------------------------------------------------------------------------

#define ALLOCATE(Rank)                                                    \
    if(playout->rank() == Rank) {                                         \
        using pimpl_type = buffer::detail_::EigenTensor<FloatType, Rank>; \
        auto ppimpl =                                                     \
          std::make_unique<pimpl_type>(playout->shape().as_smooth());     \
        return std::make_unique<buffer_type>(                             \
          std::move(ppimpl), std::move(playout), this->clone());          \
    }

TPARAMS
typename EIGEN::buffer_base_pointer EIGEN::allocate_(layout_pointer playout) {
    using buffer_type = buffer::Eigen<FloatType>;
    ALLOCATE(0)
    else ALLOCATE(1) else ALLOCATE(2) else ALLOCATE(3) else ALLOCATE(4) else ALLOCATE(5) else ALLOCATE(
      6) else ALLOCATE(7) else ALLOCATE(8) else ALLOCATE(9) else ALLOCATE(10) else {
        throw std::runtime_error("Tensors with rank > 10 not supported.");
    }
}

TPARAMS
typename EIGEN::contiguous_pointer EIGEN::construct_(rank0_il il) {
    return il_construct_(il);
}

TPARAMS
typename EIGEN::contiguous_pointer EIGEN::construct_(rank1_il il) {
    return il_construct_(il);
}

TPARAMS
typename EIGEN::contiguous_pointer EIGEN::construct_(rank2_il il) {
    return il_construct_(il);
}

TPARAMS
typename EIGEN::contiguous_pointer EIGEN::construct_(rank3_il il) {
    return il_construct_(il);
}

TPARAMS
typename EIGEN::contiguous_pointer EIGEN::construct_(rank4_il il) {
    return il_construct_(il);
}

TPARAMS
typename EIGEN::contiguous_pointer EIGEN::construct_(layout_pointer playout,
                                                     element_type value) {
    auto pbuffer        = this->allocate(std::move(playout));
    auto& contig_buffer = static_cast<buffer::Contiguous<FloatType>&>(*pbuffer);
    contig_buffer.fill(value);
    return pbuffer;
}

// -- Private

TPARAMS
template<typename ILType>
typename EIGEN::contiguous_pointer EIGEN::il_construct_(ILType il) {
    auto [extents, data] = unwrap_il(il);
    shape::Smooth shape(extents.begin(), extents.end());
    auto playout      = std::make_unique<layout::Physical>(std::move(shape));
    auto pbuffer      = this->allocate(std::move(playout));
    auto& buffer_down = rebind(*pbuffer);
    buffer_down.copy(data);
    return pbuffer;
}

#undef EIGEN
#undef TPARAMS

// -- Explicit class template instantiation

#define DEFINE_EIGEN_ALLOCATOR(TYPE) template class Eigen<TYPE>

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_EIGEN_ALLOCATOR);

#undef DEFINE_EIGEN_ALLOCATOR

} // namespace tensorwrapper::allocator
