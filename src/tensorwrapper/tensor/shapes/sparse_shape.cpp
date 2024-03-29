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

#include "detail_/sparse_shape_pimpl.hpp"

namespace tensorwrapper::tensor {

template<typename T>
using pimpl_type = detail_::SparseShapePIMPL<T>;

namespace {

template<typename FieldType>
const auto& downcast(const detail_::ShapePIMPL<FieldType>& pimpl) {
    using pimpl_t      = pimpl_type<FieldType>;
    const auto* ppimpl = dynamic_cast<const pimpl_t*>(&pimpl);
    if(ppimpl) return *ppimpl;

    // Cast failed (shouldn't be reachable unless I missed something)
    throw std::bad_cast();
}

template<typename FieldType, typename... Args>
auto make_pimpl(Args&&... args) {
    return std::make_unique<pimpl_type<FieldType>>(std::forward<Args>(args)...);
}

auto make_i2m(std::size_t i) {
    using idx2mode_type = typename SparseShape<field::Scalar>::idx2mode_type;
    idx2mode_type rv(i);
    std::iota(rv.begin(), rv.end(), 0);
    return rv;
}

} // namespace

#define SPARSE_SHAPE SparseShape<FieldType>

// Base impls
template<typename FieldType>
SPARSE_SHAPE::SparseShape(extents_type extents,
                          inner_extents_type inner_extents, sparse_map_type sm,
                          idx2mode_type i2m) :
  base_type(make_pimpl<FieldType>(std::move(extents), std::move(inner_extents),
                                  std::move(sm), std::move(i2m))) {}

template<typename FieldType>
SPARSE_SHAPE::SparseShape(tiling_type tiling, inner_extents_type inner_extents,
                          sparse_map_type sm, idx2mode_type i2m) :
  base_type(make_pimpl<FieldType>(std::move(tiling), std::move(inner_extents),
                                  std::move(sm), std::move(i2m))) {}

// Default I2M
template<typename FieldType>
SPARSE_SHAPE::SparseShape(extents_type extents,
                          inner_extents_type inner_extents,
                          sparse_map_type sm) :
  SparseShape(std::move(extents), std::move(inner_extents), std::move(sm),
              make_i2m(extents.size())) {}

template<typename FieldType>
SPARSE_SHAPE::SparseShape(tiling_type tiling, inner_extents_type inner_extents,
                          sparse_map_type sm) :
  SparseShape(std::move(tiling), std::move(inner_extents), std::move(sm),
              make_i2m(tiling.size())) {}

// Default Inner Extents
template<typename FieldType>
SPARSE_SHAPE::SparseShape(extents_type extents, sparse_map_type sm,
                          idx2mode_type i2m) :
  SparseShape(std::move(extents), inner_extents_type{}, std::move(sm),
              std::move(i2m)) {}

template<typename FieldType>
SPARSE_SHAPE::SparseShape(tiling_type tiling, sparse_map_type sm,
                          idx2mode_type i2m) :
  SparseShape(std::move(tiling), inner_extents_type{}, std::move(sm),
              std::move(i2m)) {}

// Default Inner Extents + I2M
template<typename FieldType>
SPARSE_SHAPE::SparseShape(extents_type extents, sparse_map_type sm) :
  SparseShape(std::move(extents), inner_extents_type{}, std::move(sm)) {}

template<typename FieldType>
SPARSE_SHAPE::SparseShape(tiling_type tiling, sparse_map_type sm) :
  SparseShape(std::move(tiling), inner_extents_type{}, std::move(sm)) {}

// Copy
template<typename FieldType>
SPARSE_SHAPE::SparseShape(const SparseShape& other) :
  base_type(other.has_pimpl_() ?
              make_pimpl<FieldType>(downcast(other.pimpl_())) :
              typename base_type::pimpl_pointer{}) {}

template<typename FieldType>
bool SPARSE_SHAPE::operator==(const SparseShape& rhs) const noexcept {
    if(!this->has_pimpl_() && !rhs.has_pimpl_()) return true;
    if(this->has_pimpl_() && rhs.has_pimpl_())
        return downcast(this->pimpl_()) == downcast(rhs.pimpl_());
    return false;
}

template<typename FieldType>
typename SPARSE_SHAPE::const_sparse_map_reference SPARSE_SHAPE::sparse_map()
  const {
    return downcast(this->pimpl_()).sparse_map();
}

template<typename FieldType>
typename SPARSE_SHAPE::const_idx2mode_reference SPARSE_SHAPE::idx2mode_map()
  const {
    return downcast(this->pimpl_()).idx2mode_map();
}

//------------------------------------------------------------------------------
//                   Protected/Private Member Functions
//------------------------------------------------------------------------------

template<typename FieldType>
bool SPARSE_SHAPE::is_hard_zero_(const index_type& i) const {
    return downcast(this->pimpl_()).is_hard_zero(i);
}

template<typename FieldType>
bool SPARSE_SHAPE::is_hard_zero_(const index_type& lo,
                                 const index_type& hi) const {
    return downcast(this->pimpl_()).is_hard_zero(lo, hi);
}

template<typename FieldType>
typename SPARSE_SHAPE::pointer_type SPARSE_SHAPE::slice_(
  const index_type& lo, const index_type& hi) const {
    auto pimpl_ptr = downcast(this->pimpl_()).slice(lo, hi);

    return pointer_type(new SparseShape(
      pimpl_ptr->extents(), pimpl_ptr->inner_extents(),
      downcast(*pimpl_ptr).sparse_map(), downcast(*pimpl_ptr).idx2mode_map()));
}

template<typename FieldType>
typename SPARSE_SHAPE::pointer_type SPARSE_SHAPE::clone_() const {
    return pointer_type(new SparseShape(*this));
}

template<typename FieldType>
bool SPARSE_SHAPE::is_equal_(const Shape<FieldType>& rhs) const noexcept {
    using const_pointer_type = const SparseShape<FieldType>*;
    auto prhs                = dynamic_cast<const_pointer_type>(&rhs);

    // If null, rhs does not contain a SparseShape part
    if(!prhs) return false;

    // Not null, dereference and compare SparseShape parts
    return *this == *prhs;
}

#undef SPARSE_SHAPE

template class SparseShape<field::Scalar>;
template class SparseShape<field::Tensor>;

} // namespace tensorwrapper::tensor
