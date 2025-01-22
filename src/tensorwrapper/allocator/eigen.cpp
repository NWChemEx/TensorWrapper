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

#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/detail_/unique_ptr_utilities.hpp>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::allocator {
namespace {
template<typename EigenTensorType, typename ShapeType, std::size_t... Is>
auto unwrap_shape(const ShapeType& shape, std::index_sequence<Is...>) {
    // XXX: This is a hack until we have a general Shape API in place
    auto const_shape = static_cast<const shape::Smooth&>(shape);
    return EigenTensorType(const_shape.extent(Is)...);
}

} // namespace

#define TPARAMS template<typename FloatType, unsigned short Rank>
#define EIGEN Eigen<FloatType, Rank>

TPARAMS
typename EIGEN::eigen_buffer_pointer EIGEN::allocate(
  eigen_layout_pointer playout) {
    using eigen_data_type = typename eigen_buffer_type::data_type;
    if(playout->shape().rank() != Rank)
        throw std::runtime_error("Rank of the layout is not compatible");

    return std::make_unique<eigen_buffer_type>(
      unwrap_shape<eigen_data_type>(playout->shape(),
                                    std::make_index_sequence<Rank>()),
      *playout);
}

TPARAMS
bool EIGEN::can_rebind(const_buffer_base_reference buffer) {
    auto pbuffer = dynamic_cast<const eigen_buffer_type*>(&buffer);
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

#define ALLOCATE_CONDITION(RANK) \
    if(rank == RANK) return std::make_unique<Eigen<FloatType, RANK>>(rv)

TPARAMS
typename EIGEN::base_pointer EIGEN::make_eigen_allocator(unsigned int rank,
                                                         runtime_view_type rv) {
    ALLOCATE_CONDITION(0);
    else ALLOCATE_CONDITION(1);
    else ALLOCATE_CONDITION(2);
    else ALLOCATE_CONDITION(3);
    else ALLOCATE_CONDITION(4);
    else ALLOCATE_CONDITION(5);
    else ALLOCATE_CONDITION(6);
    else ALLOCATE_CONDITION(7);
    else ALLOCATE_CONDITION(8);
    else ALLOCATE_CONDITION(9);
    else ALLOCATE_CONDITION(10);
    throw std::runtime_error(
      "Presently only support eigen tensors up to rank 10");
}

#undef ALLOCATE_CONDITION

// -----------------------------------------------------------------------------
// -- Protected methods
// -----------------------------------------------------------------------------

TPARAMS
typename EIGEN::buffer_base_pointer EIGEN::allocate_(layout_pointer playout) {
    auto pderived = detail_::dynamic_pointer_cast<eigen_layout_type>(playout);
    if(pderived == nullptr) throw std::runtime_error("Unsupported layout");

    return allocate(std::move(pderived));
}

#undef EIGEN
#undef TPARAMS

// -- Explicit class template instantiation

#define DEFINE_EIGEN_ALLOCATOR(TYPE)        \
    template class Eigen<TYPE, 0>; \
    template class Eigen<TYPE, 1>; \
    template class Eigen<TYPE, 2>; \
    template class Eigen<TYPE, 3>; \
    template class Eigen<TYPE, 4>; \
    template class Eigen<TYPE, 5>; \
    template class Eigen<TYPE, 6>; \
    template class Eigen<TYPE, 7>; \
    template class Eigen<TYPE, 8>; \
    template class Eigen<TYPE, 9>; \
    template class Eigen<TYPE, 10>

DEFINE_EIGEN_ALLOCATOR(float);
DEFINE_EIGEN_ALLOCATOR(double);

#ifdef ENABLE_SIGMA
DEFINE_EIGEN_ALLOCATOR(sigma::UFloat);
DEFINE_EIGEN_ALLOCATOR(sigma::UDouble);
#endif

#undef DEFINE_EIGEN_ALLOCATOR

} // namespace tensorwrapper::allocator
