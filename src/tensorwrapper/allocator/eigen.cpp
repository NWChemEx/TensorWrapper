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

TPARAMS
typename EIGEN::buffer_base_pointer EIGEN::allocate_(layout_pointer playout) {
    using eigen_data_type = typename eigen_buffer_type::data_type;
    if(playout->shape().rank() != Rank) {
        auto palloc =
          make_eigen_allocator(playout->shape().rank(), this->runtime());
        return palloc->allocate(std::move(playout));
    }

    return std::make_unique<eigen_buffer_type>(
      unwrap_shape<eigen_data_type>(playout->shape(),
                                    std::make_index_sequence<Rank>()),
      *playout, *this);
}

TPARAMS
typename EIGEN::buffer_base_pointer EIGEN::construct_(layout_pointer playout,
                                                      element_type value) {
    auto pbuffer        = this->allocate(std::move(playout));
    auto& contig_buffer = static_cast<buffer::Contiguous<FloatType>&>(*pbuffer);
    auto* pdata         = contig_buffer.data();
    std::fill(pdata, pdata + contig_buffer.size(), value);
    return pbuffer;
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::addition_assignment_(
  label_type this_labels, const_labeled_reference, const_labeled_reference) {
    return make_eigen_allocator(this_labels.size(), this->runtime());
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::subtraction_assignment_(
  label_type this_labels, const_labeled_reference, const_labeled_reference) {
    return make_eigen_allocator(this_labels.size(), this->runtime());
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::multiplication_assignment_(
  label_type this_labels, const_labeled_reference, const_labeled_reference) {
    return make_eigen_allocator(this_labels.size(), this->runtime());
}

TPARAMS
typename EIGEN::dsl_reference EIGEN::permute_assignment_(
  label_type this_labels, const_labeled_reference) {
    return make_eigen_allocator(this_labels.size(), this->runtime());
}

#undef EIGEN
#undef TPARAMS

// -- Explicit class template instantiation

#define DEFINE_EIGEN_ALLOCATOR(TYPE) template class Eigen<TYPE>

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_EIGEN_ALLOCATOR);

#undef DEFINE_EIGEN_ALLOCATOR

} // namespace tensorwrapper::allocator
