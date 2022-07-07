#include "tensorwrapper/tensor/detail_/pimpl.hpp"
/// Used in pimpl_ for default value
#include "./buffer/detail_/ta_buffer_pimpl.hpp"
/// Used for initializer list construction
#include "./detail_/ta_to_tw.hpp"

namespace tensorwrapper::tensor {

namespace {
#if 1

template<typename ElementType, std::size_t Rank, typename FieldType>
auto il_to_tw(const n_d_initializer_list_t<ElementType, Rank>& il) {
    using traits_type      = detail_::FieldTraits<FieldType>;
    using variant_type     = typename traits_type::variant_type;
    using default_tensor_t = std::variant_alternative_t<0, variant_type>;

    auto& world = TA::get_default_world();

    if constexpr(std::is_same_v<FieldType, field::Scalar>) {
        return std::move(detail_::ta_to_tw(default_tensor_t(world, il)));
    } else {
        throw std::runtime_error("ToT initializer lists NYI.");
        return TensorWrapper<FieldType>();
    }
}

#if 0

template<typename VariantType>
auto new_variant(const VariantType& other,
                 const SparseShape<field::Tensor>& shape,
                 const Allocator<field::Tensor>& alloc) {
    const auto sm       = shape.sparse_map();
    const auto tr       = alloc.make_tiled_range(shape.extents());
    const auto idx2mode = shape.idx2mode_map();
    std::map<std::size_t, std::size_t> idx2mode_map;
    for(std::size_t i = 0; i < idx2mode.size(); ++i)
        idx2mode_map[i] = idx2mode[i];
    using variant_type =
      typename detail_::FieldTraits<field::Tensor>::variant_type;
    auto l = [=](auto&& tensor_in) {
        return variant_type{from_sparse_map(sm, tensor_in, tr, idx2mode_map)};
    };
    return std::visit(l, other);
}

#endif
} // namespace

// Macro to avoid typing the full type of the TensorWrapper
#define TENSOR_WRAPPER TensorWrapper<FieldType>

//------------------------------------------------------------------------------
//                            Ctors
//------------------------------------------------------------------------------

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper() = default;

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(pimpl_pointer p) noexcept :
  m_pimpl_(std::move(p)) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(const tile_populator_type& fxn, shape_pointer s,
                              allocator_pointer a) :
  TensorWrapper(std::make_unique<pimpl_type>(a->allocate(fxn, *s), std::move(s),
                                             std::move(a))) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(const element_populator_type& fxn,
                              shape_pointer s, allocator_pointer a) :
  TensorWrapper(std::make_unique<pimpl_type>(a->allocate(fxn, *s), std::move(s),
                                             std::move(a))) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(buffer_type buffer, shape_pointer shape,
                              allocator_pointer alloc) :
  TensorWrapper(std::make_unique<pimpl_type>(
    std::make_unique<buffer_type>(std::move(buffer)), std::move(shape),
    std::move(alloc))) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(allocator_pointer p) :
  TensorWrapper(buffer_type{}, shape_pointer{}, std::move(p)) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(shape_pointer shape, allocator_pointer p) :
  TensorWrapper(buffer_type{}, std::move(shape), std::move(p)) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(n_d_initializer_list_t<element_type, 1> il) :
  TensorWrapper(il_to_tw<element_type, 1, FieldType>(il)) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(n_d_initializer_list_t<element_type, 2> il) :
  TensorWrapper(il_to_tw<element_type, 2, FieldType>(il)) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(n_d_initializer_list_t<element_type, 3> il) :
  TensorWrapper(il_to_tw<element_type, 3, FieldType>(il)) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(n_d_initializer_list_t<element_type, 4> il) :
  TensorWrapper(il_to_tw<element_type, 4, FieldType>(il)) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(const TensorWrapper& other) :
  m_pimpl_(other.m_pimpl_ ? other.pimpl_().clone() : nullptr) {}

template<typename FieldType>
TENSOR_WRAPPER::TensorWrapper(TensorWrapper&& other) = default;

template<typename FieldType>
TENSOR_WRAPPER& TENSOR_WRAPPER::operator=(const TensorWrapper& other) {
    if(this == &other) return *this;

    if(other.m_pimpl_)
        other.pimpl_().clone().swap(m_pimpl_);
    else
        m_pimpl_.reset();

    return *this;
}

template<typename FieldType>
TENSOR_WRAPPER& TENSOR_WRAPPER::operator=(TensorWrapper&& other) = default;

template<typename FieldType>
TENSOR_WRAPPER::~TensorWrapper() noexcept = default;

template<typename FieldType>
void TENSOR_WRAPPER::swap(TensorWrapper& other) noexcept {
    m_pimpl_.swap(other.m_pimpl_);
}

//------------------------------------------------------------------------------
//                           Accessors
//------------------------------------------------------------------------------

template<typename FieldType>
typename TENSOR_WRAPPER::const_allocator_reference TENSOR_WRAPPER::allocator()
  const {
    return pimpl_().allocator();
}

template<typename FieldType>
void TENSOR_WRAPPER::reallocate(allocator_pointer p) {
    pimpl_().reallocate(std::move(p));
}

template<typename FieldType>
typename TENSOR_WRAPPER::labeled_tensor_type TENSOR_WRAPPER::operator()(
  const annotation_type& annotation) {
    return labeled_tensor_type(annotation, *this);
}

template<typename FieldType>
typename TENSOR_WRAPPER::const_labeled_tensor_type TENSOR_WRAPPER::operator()(
  const annotation_type& annotation) const {
    return const_labeled_tensor_type(annotation, *this);
}

template<typename FieldType>
typename TENSOR_WRAPPER::annotation_type TENSOR_WRAPPER::make_annotation(
  const annotation_type& letter) const {
    return m_pimpl_ ? pimpl_().make_annotation(letter) : "";
}

template<typename FieldType>
typename TENSOR_WRAPPER::rank_type TENSOR_WRAPPER::rank() const {
    if(m_pimpl_) return pimpl_().rank();
    return rank_type{0};
}

template<typename FieldType>
typename TENSOR_WRAPPER::extents_type TENSOR_WRAPPER::extents() const {
    if(m_pimpl_) return pimpl_().extents();
    return extents_type{};
}

template<typename FieldType>
typename TENSOR_WRAPPER::size_type TENSOR_WRAPPER::size() const {
    if(m_pimpl_) return pimpl_().size();
    return size_type{0};
}

template<typename FieldType>
typename TENSOR_WRAPPER::const_shape_reference TENSOR_WRAPPER::shape() const {
    return pimpl_().shape();
}

template<typename FieldType>
TENSOR_WRAPPER TENSOR_WRAPPER::slice(const il_type& lo, const il_type& hi,
                                     allocator_pointer p) const {
    TENSOR_WRAPPER rv;
    rv.m_pimpl_ = pimpl_().slice(lo, hi, std::move(p));
    return rv;
}

template<typename FieldType>
TENSOR_WRAPPER TENSOR_WRAPPER::reshape(shape_pointer shape) const {
    TENSOR_WRAPPER rv(*this);
    rv.pimpl_().reshape(std::move(shape));
    return rv;
}

template<typename FieldType>
typename TENSOR_WRAPPER::scalar_value_type TENSOR_WRAPPER::norm() const {
    return pimpl_().norm();
}

template<typename FieldType>
typename TENSOR_WRAPPER::scalar_value_type TENSOR_WRAPPER::sum() const {
    return pimpl_().sum();
}

template<typename FieldType>
typename TENSOR_WRAPPER::scalar_value_type TENSOR_WRAPPER::trace() const {
    return pimpl_().trace();
}

template<typename FieldType>
std::ostream& TENSOR_WRAPPER::print(std::ostream& os) const {
    if(!m_pimpl_) return os;
    return pimpl_().print(os);
}

template<typename FieldType>
void TENSOR_WRAPPER::hash(tensorwrapper::detail_::Hasher& h) const {
    if(m_pimpl_) pimpl_().hash(h);
}

template<typename FieldType>
bool TENSOR_WRAPPER::operator==(const TensorWrapper& rhs) const {
    if(m_pimpl_ && rhs.m_pimpl_)
        return pimpl_() == rhs.pimpl_();
    else if(!m_pimpl_ && !rhs.m_pimpl_)
        return true;
    return false;
}

template<typename FieldType>
typename TENSOR_WRAPPER::pimpl_reference TENSOR_WRAPPER::pimpl() {
    if(!m_pimpl_) throw std::runtime_error("No TW PIMPL");
    return *m_pimpl_;
}

template<typename FieldType>
typename TENSOR_WRAPPER::buffer_reference TENSOR_WRAPPER::buffer() {
    return pimpl_().buffer();
}

template<typename FieldType>
typename TENSOR_WRAPPER::const_buffer_reference TENSOR_WRAPPER::buffer() const {
    return pimpl_().buffer();
}

//------------------------------------------------------------------------------
//                  Protected and Private Members
//------------------------------------------------------------------------------

template<typename FieldType>
typename TENSOR_WRAPPER::variant_type& TENSOR_WRAPPER::variant_() {
    return pimpl_().variant();
}

template<typename FieldType>
const typename TENSOR_WRAPPER::variant_type& TENSOR_WRAPPER::variant_() const {
    return pimpl_().variant();
}

template<typename FieldType>
typename TENSOR_WRAPPER::labeled_variant_type TENSOR_WRAPPER::annotate_(
  const annotation_type& annotation) {
    return pimpl_().annotate(annotation);
}

template<typename FieldType>
typename TENSOR_WRAPPER::const_labeled_type TENSOR_WRAPPER::annotate_(
  const annotation_type& annotation) const {
    return pimpl_().annotate(annotation);
}

template<typename FieldType>
typename TENSOR_WRAPPER::pimpl_reference TENSOR_WRAPPER::pimpl_() {
    if(!m_pimpl_) {
        using ta_pimpl_type  = buffer::detail_::TABufferPIMPL<FieldType>;
        using inner_ext_type = typename shape_type::inner_extents_type;

        /// Inner extents can't just be defaulted for ToT cases
        inner_ext_type inner_ext{};
        if constexpr(std::is_same_v<FieldType, field::Tensor>) {
            /// Pseudo-default value
            inner_ext[index_type{0}] = Shape<field::Scalar>(extents_type{0});
        }

        auto pt  = std::make_unique<ta_pimpl_type>();
        m_pimpl_ = std::make_unique<pimpl_type>(
          std::make_unique<buffer_type>(std::move(pt)),
          std::make_unique<shape_type>(extents_type{0}, inner_ext),
          default_allocator<field_type>());
    }
    return *m_pimpl_;
}

template<typename FieldType>
typename TENSOR_WRAPPER::const_pimpl_reference TENSOR_WRAPPER::pimpl_() const {
    if(m_pimpl_) return *m_pimpl_;
    throw std::runtime_error(
      "Tensor has no PIMPL. Was it default constructed or moved from?");
}

template<typename FieldType>
void TENSOR_WRAPPER::update_shape_() {
    if(m_pimpl_) m_pimpl_->update_shape();
}

#undef TENSOR_WRAPPER

// template TensorWrapper<field::Tensor>::TensorWrapper<field::Scalar, void>(
//  const TensorWrapper<field::Scalar>&,
//  typename TensorWrapper<field::Tensor>::sparse_pointer,
//  typename TensorWrapper<field::Tensor>::allocator_pointer);

template class TensorWrapper<field::Scalar>;
template class TensorWrapper<field::Tensor>;
#endif
} // namespace tensorwrapper::tensor
