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

#pragma once
#include "../../sparse_map/sparse_map/detail_/tiling_map_index.hpp"
#include "../buffer/detail_/ta_buffer_pimpl.hpp"
#include "tensorwrapper/tensor/allocator/tiled_array.hpp"

namespace tensorwrapper::tensor::allocator::detail_ {

using tiled_range_type = TA::TiledRange;
using tr1_type         = TA::TiledRange1;

using sparse_shape_type = SparseShape<field::Scalar>;
using sparse_map_type   = sparse_shape_type::sparse_map_type;
using idx2mode_type     = sparse_shape_type::idx2mode_type;

using size_type  = unsigned int;
using index_type = std::vector<size_type>;
using tile_index = tensorwrapper::sparse_map::Index;

using ta_shape_type = TA::SparseShape<float>;

template<typename FieldType>
using ta_buffer_pimpl_type =
  tensorwrapper::tensor::buffer::detail_::TABufferPIMPL<FieldType>;
template<typename FieldType>
using default_tensor_type =
  typename ta_buffer_pimpl_type<FieldType>::default_tensor_type;
template<typename FieldType>
using lazy_tensor_type =
  typename ta_buffer_pimpl_type<FieldType>::lazy_tensor_type;

} // namespace tensorwrapper::tensor::allocator::detail_
