#include "detail_/sparse_shape_pimpl.hpp"

namespace tensorwrapper::tensor::novel {

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

template<typename FieldType>
SPARSE_SHAPE::SparseShape(extents_type extents, sparse_map_type sm,
                          inner_extents_type inner_extents) :
  SparseShape(std::move(extents), std::move(sm), make_i2m(extents.size()),
              std::move(inner_extents)) {}

template<typename FieldType>
SPARSE_SHAPE::SparseShape(extents_type extents, sparse_map_type sm,
                          idx2mode_type i2m, inner_extents_type inner_extents) :
  base_type(make_pimpl<FieldType>(std::move(extents), std::move(sm),
                                  std::move(i2m), std::move(inner_extents))) {}

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
