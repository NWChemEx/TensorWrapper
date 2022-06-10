#include "../../../ta_helpers/slice.hpp"
#include "../../../ta_helpers/ta_helpers.hpp"
#include "tensorwrapper/tensor/novel/detail_/pimpl.hpp"

#include "../../buffer/detail_/ta_buffer_pimpl.hpp"
#include "../shapes/detail_/sparse_shape_pimpl.hpp"

namespace tensorwrapper::tensor::novel::detail_ {
namespace {

// TODO: This is basically tensorwrapper::tensor::to_vector copy/pasted, need to
//       work this into main to_vector routine
auto to_vector_from_pimpl(const TensorWrapperPIMPL<field::Scalar>& t) {
    auto t_ta = std::get<TA::TSpArrayD>(t.variant());
    t_ta.make_replicated();
    std::vector<double> rv(t.size(), 0.0);

    for(const auto& tile_i : t_ta) {
        const auto& i_range = tile_i.get().range();
        for(auto idx : i_range) rv[i_range.ordinal(idx)] = tile_i.get()[idx];
    }
    return rv;
}

#if 0
template<typename VariantType>
auto make_extents(VariantType&& v) {
    using extents_type = TensorWrapperPIMPL<field::Scalar>::extents_type;
    using size_type    = typename extents_type::size_type;

    auto l = [=](auto&& t) {
        if(!t.is_initialized()) return extents_type{};
        const auto& tr = t.trange();
        extents_type rv(tr.rank());
        const auto& erange = tr.elements_range().extent();
        for(size_type i = 0; i < rv.size(); ++i) rv[i] = erange[i];
        return rv;
    };
    return std::visit(l, std::forward<VariantType>(v));
}

#endif

// TODO: This should live in Buffer, but can't until new TW
// infrastructure replaces old
template<typename FieldType>
void reshape_helper(buffer::Buffer<FieldType>& buffer,
                    const Shape<FieldType>& shape) {
    auto l = [&](auto&& old_tensor) {
        TA::foreach_inplace(old_tensor, [&](auto&& tile) {
            const auto range = tile.range();
            const auto lo    = range.lobound();
            const auto up    = range.upbound();
            sparse_map::Index lo_idx(lo.begin(), lo.end());
            sparse_map::Index up_idx(up.begin(), up.end());
            if(shape.is_hard_zero(lo_idx, up_idx)) { tile.scale_to(0.); }
            return TA::norm(tile);
        });
    };
    std::visit(l, buffer.variant());
}

/// XXX This should be replaced with Buffer::slice
template<typename FieldType>
auto slice_helper(buffer::Buffer<FieldType>& buffer,
                  const sparse_map::Index& low, const sparse_map::Index& high) {
    using ta_pimpl_type =
      tensorwrapper::tensor::buffer::detail_::TABufferPIMPL<FieldType>;
    auto* ta_pimpl = dynamic_cast<ta_pimpl_type*>(buffer.pimpl());
    if(!ta_pimpl)
        throw std::runtime_error("Slice only implemented for TA Backends");

    auto l = [=](auto&& arg) {
        using clean_t         = std::decay_t<decltype(arg)>;
        constexpr bool is_tot = TensorTraits<clean_t>::is_tot;
        clean_t rv;
        if constexpr(is_tot) {
            throw std::runtime_error("Can't slice a ToT.");
        } else {
            rv = ta_helpers::slice(arg, low, high);
        }
        return rv;
    };

    auto slice_pimpl =
      std::make_unique<ta_pimpl_type>(std::visit(l, buffer.variant()));

    return std::make_unique<buffer::Buffer<FieldType>>(std::move(slice_pimpl));
}
} // namespace

// Macro to avoid retyping the full type of the PIMPL
#define PIMPL_TYPE TensorWrapperPIMPL<FieldType>

template<typename FieldType>
PIMPL_TYPE::TensorWrapperPIMPL(buffer_pointer b, shape_pointer s,
                               allocator_pointer p) :
  m_buffer_(std::move(b)), m_allocator_(std::move(p)), m_shape_(std::move(s)) {}

template<typename FieldType>
typename PIMPL_TYPE::pimpl_pointer PIMPL_TYPE::clone() const {
    allocator_pointer new_alloc(m_allocator_ ? m_allocator_->clone() : nullptr);
    shape_pointer new_shape(m_shape_ ? m_shape_->clone() : nullptr);
    buffer_pointer new_buffer(
      m_buffer_ ? std::make_unique<buffer_type>(*m_buffer_) : nullptr);
    return std::make_unique<my_type>(
      std::move(new_buffer), std::move(new_shape), std::move(new_alloc));
}

template<typename FieldType>
typename PIMPL_TYPE::const_allocator_reference PIMPL_TYPE::allocator() const {
    if(m_allocator_) return *m_allocator_;
    throw std::runtime_error("Tensor has no allocator!!!!");
}

template<typename FieldType>
typename PIMPL_TYPE::const_shape_reference PIMPL_TYPE::shape() const {
    if(m_shape_) return *m_shape_;
    throw std::runtime_error("Tensor has no shape!!!!");
}

template<typename FieldType>
typename PIMPL_TYPE::const_buffer_reference PIMPL_TYPE::buffer() const {
    if(m_buffer_) return *m_buffer_;
    throw std::runtime_error("Tensor has no buffer!!!!");
}

template<typename FieldType>
typename PIMPL_TYPE::buffer_reference PIMPL_TYPE::buffer() {
    if(m_buffer_) return *m_buffer_;
    throw std::runtime_error("Tensor has no buffer!!!!");
}

template<typename FieldType>
typename PIMPL_TYPE::labeled_variant_type PIMPL_TYPE::annotate(
  const annotation_type& annotation) {
    auto& m_tensor_   = buffer().variant();
    using new_variant = labeled_variant_t<variant_type>;
    auto l            = [&](auto&& t) { return new_variant(t(annotation)); };
    return std::visit(l, m_tensor_);
}

template<typename FieldType>
typename PIMPL_TYPE::const_labeled_type PIMPL_TYPE::annotate(
  const annotation_type& annotation) const {
    auto& m_tensor_   = buffer().variant();
    using new_variant = const_labeled_type;
    auto l            = [&](auto&& t) { return new_variant(t(annotation)); };
    return std::visit(l, m_tensor_);
}

template<typename FieldType>
typename PIMPL_TYPE::extents_type PIMPL_TYPE::extents() const {
    if(m_shape_) {
        auto ex = m_shape_->extents();
        if(ex != extents_type{}) return ex;
    }
    return extents_type{};
}

template<typename FieldType>
typename PIMPL_TYPE::annotation_type PIMPL_TYPE::make_annotation(
  const annotation_type& letter) const {
    auto r                = rank();
    constexpr bool is_tot = std::is_same_v<FieldType, field::Tensor>;
    auto outer_rank       = (is_tot ? outer_rank_() : r);
    annotation_type x;
    if(r == 0) return x;
    for(decltype(r) i = 0; i < r - 1; ++i) {
        x += letter + std::to_string(i);
        x += (i + 1 == outer_rank ? ";" : ",");
    }
    x += letter + std::to_string(r - 1);
    return x;
}

/// XXX THESE ARE TO BE REMOVED
template<typename FieldType>
typename PIMPL_TYPE::variant_type& PIMPL_TYPE::variant() {
    return buffer().variant();
}

/// XXX THESE ARE TO BE REMOVED
template<typename FieldType>
const typename PIMPL_TYPE::variant_type& PIMPL_TYPE::variant() const {
    return buffer().variant();
}

template<typename FieldType>
typename PIMPL_TYPE::rank_type PIMPL_TYPE::rank() const {
    return outer_rank_() + inner_rank_();
}

template<typename FieldType>
void PIMPL_TYPE::reallocate(allocator_pointer p) {
    reallocate_(*p);
    m_allocator_ = std::move(p);
}

template<typename FieldType>
void PIMPL_TYPE::reshape(shape_pointer pshape) {
    reshape_(*pshape);
    m_shape_ = std::move(pshape);
}

template<typename FieldType>
typename PIMPL_TYPE::scalar_value_type PIMPL_TYPE::norm() const {
    return buffer().norm();
}

template<typename FieldType>
typename PIMPL_TYPE::scalar_value_type PIMPL_TYPE::sum() const {
    return buffer().sum();
}

template<typename FieldType>
typename PIMPL_TYPE::scalar_value_type PIMPL_TYPE::trace() const {
    return buffer().trace();
}

template<typename FieldType>
typename PIMPL_TYPE::size_type PIMPL_TYPE::size() const {
    auto dims = extents();
    if(dims.empty()) return 0;
    std::multiplies<size_type> fxn;
    return std::accumulate(dims.begin(), dims.end(), size_type{1}, fxn);
}

template<typename FieldType>
typename PIMPL_TYPE::pimpl_pointer PIMPL_TYPE::slice(
  const il_type& lo, const il_type& hi, allocator_pointer p) const {
#if 0
    sparse_map::Index low(lo), high(hi);
    auto l = [=](auto&& arg) {
        using clean_t         = std::decay_t<decltype(arg)>;
        constexpr bool is_tot = TensorTraits<clean_t>::is_tot;
        clean_t rv;
        if constexpr(is_tot) {
            throw std::runtime_error("Can't slice a ToT.");
        } else {
            rv = ta_helpers::slice(arg, low, high);
        }
        return rv;
    };

    return std::make_unique<my_type>(std::visit(l, m_tensor_), std::move(p));
#else
    // throw std::runtime_error("TWPIMPL::slice NYI");
    // return nullptr;
    if(!p or !m_allocator_->is_equal(*p))
        throw std::runtime_error("slice + reallocate NYI");
    return std::make_unique<my_type>(slice_helper(*m_buffer_, lo, hi),
                                     m_shape_->slice(lo, hi), std::move(p));
#endif
}

template<typename FieldType>
std::ostream& PIMPL_TYPE::print(std::ostream& os) const {
    os << *m_buffer_;
    return os;
}

template<typename FieldType>
void PIMPL_TYPE::hash(tensorwrapper::detail_::Hasher& h) const {
    h(m_shape_, m_allocator_, m_buffer_);
}

template<typename FieldType>
bool PIMPL_TYPE::operator==(const TensorWrapperPIMPL& rhs) const {
    // Compare shapes
    if(m_shape_ && rhs.m_shape_) {
        if(!m_shape_->is_equal(*rhs.m_shape_)) return false;
    } else if(!m_shape_ != !rhs.m_shape_)
        return false;

    // Compare allocators
    if(m_allocator_ && rhs.m_allocator_) {
        if(!m_allocator_->is_equal(*rhs.m_allocator_)) return false;
    } else if(!m_allocator_ != !rhs.m_allocator_)
        return false;

    // Compare buffers
    if(m_buffer_ && rhs.m_buffer_) {
        return *m_buffer_ == *rhs.m_buffer_;
    } else
        return m_buffer_ == rhs.m_buffer_;
}

template<typename FieldType>
void PIMPL_TYPE::update_shape() {
#if 0
    auto new_shape = std::make_unique<shape_type>(make_extents(m_tensor_));
    if(m_shape_ && extents() == new_shape->extents()) return;
    m_shape_.swap(new_shape);
#else
    throw std::runtime_error("TWPIMPL::update_shape NYI");
#endif
}

//------------------------------------------------------------------------------
//                     Private Member Functions
//------------------------------------------------------------------------------

template<typename FieldType>
void PIMPL_TYPE::reshape_(const shape_type& other) {
#if 0
    auto shape = other.extents();

    // Short-circuit if shapes are polymorphically equivalent
    if(m_shape_->is_equal(other)) return;

    // If the extents aren't the same we're shuffling elements around
    if(shape != extents()) shuffle_(shape);

    // TODO: This is basically a hack to compare the sparsities
    auto ta_tensor = other.make_tensor(*m_allocator_);

    auto l = [&](auto&& old_tensor) {
        auto m = [&](auto&& new_tensor) {
            if(old_tensor.shape() == new_tensor.shape()) return;
            const auto& new_shape = new_tensor.shape();
            auto dummy_idx        = make_annotation("j");
            old_tensor(dummy_idx) = old_tensor(dummy_idx).set_shape(new_shape);
        };
        std::visit(m, ta_tensor);
    };
    std::visit(l, m_tensor_);
#else
    // Short-circuit if shapes are polymorphically equivalent
    if(m_shape_->is_equal(other)) return;

    // If the extents aren't the same we're shuffling elements around
    if(m_shape_->extents() != other.extents()) shuffle_(other);

    // Apply sparsity
    reshape_helper(*m_buffer_, other);
#endif
}

template<typename FieldType>
void PIMPL_TYPE::reallocate_(const_allocator_reference p) {
#if 0
    auto l = [&](auto&& arg) {
        // We have nothing to do if it's not initialized yet
        if(!arg.is_initialized()) return;

        // Only retile if the tiled ranges are different
        const auto tr = p.make_tiled_range(extents());
        if(arg.trange() != tr) {
            if constexpr(std::is_same_v<FieldType, field::Scalar>) {
                arg = TA::retile(arg, tr);
            } else {
                throw std::runtime_error("reallocate for ToT NYI!!!");
            }
        }
    };
    std::visit(l, m_tensor_);
#else
    if(m_allocator_ and m_shape_) {
        m_buffer_ = p.reallocate(*m_buffer_, *m_shape_);
    }
#endif
}

template<typename FieldType>
void PIMPL_TYPE::shuffle_(const shape_type& shape) {
    const auto times_op = std::multiplies<size_t>();
    auto extents        = shape.extents();
    auto new_volume =
      std::accumulate(extents.begin(), extents.end(), 1, times_op);

    if(new_volume != size()) {
        std::string msg =
          "Volume of the new shape: " + std::to_string(new_volume) +
          " is not the same as " +
          "the volume of the old shape: " + std::to_string(size());
        throw std::runtime_error(msg);
    }

#if 0
    auto tr = m_allocator_->make_tiled_range(shape);

    // TODO: Use a distribution aware algorithm
    auto l = [=](auto&& arg) {
        using clean_t = std::decay_t<decltype(arg)>;
        clean_t rv;
        if constexpr(TensorTraits<clean_t>::is_tot) {
            std::runtime_error("Can't reshape a ToT");
        } else {
            auto data = to_vector_from_pimpl(*this);
            rv        = TA::make_array<clean_t>(
              arg.world(), tr, [=](auto& tile, const auto& range) {
                  tile = std::decay_t<decltype(tile)>(range);
                  for(const auto& new_idx : range) {
                      tile[new_idx] = data[range.ordinal(new_idx)];
                  }
                  return tile.norm();
              });
        }
        return rv;
    };
    m_tensor_ = std::visit(l, m_tensor_);
#else
    if constexpr(field::is_scalar_field_v<FieldType>) {
        auto data   = to_vector_from_pimpl(*this);
        size_t rank = shape.extents().size();
        std::vector<size_t> stride_data(rank);
        {
            size_t _vol = 1;
            for(int d = rank - 1; d >= 0; d--) {
                stride_data[d] = _vol;
                _vol *= shape.extents()[d];
            }
        }
        m_buffer_ = m_allocator_->allocate(
          [=, d = std::move(data),
           s = std::move(stride_data)](auto idx) -> double {
              size_t ordinal = 0;
              for(int i = 0; i < rank; ++i) ordinal += s[i] * idx[i];
              return d[ordinal];
          },
          shape);
    } else {
        throw std::runtime_error("TW:shuffle_ for ToT NYI");
    }
    m_shape_ = shape.clone();
#endif
}

template<typename FieldType>
typename PIMPL_TYPE::rank_type PIMPL_TYPE::inner_rank_() const {
    if constexpr(field::is_tensor_field_v<FieldType>) {
        if(!m_shape_) return 0;
        if(!m_shape_->inner_extents().size()) return 0;
        auto& [idx, inner_shape] = *m_shape_->inner_extents().begin();
        return inner_shape.extents().size();
    } else
        return 0;
}

template<typename FieldType>
typename PIMPL_TYPE::rank_type PIMPL_TYPE::outer_rank_() const noexcept {
    return m_shape_ ? m_shape_->extents().size() : 0;
}

#undef PIMPL_TYPE

template class TensorWrapperPIMPL<field::Scalar>;
template class TensorWrapperPIMPL<field::Tensor>;

} // namespace tensorwrapper::tensor::novel::detail_
