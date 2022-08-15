#include "ta_buffer_pimpl.hpp"

#include "../../../ta_helpers/einsum/einsum.hpp"
#include "../../../ta_helpers/ta_helpers.hpp"

#include <utilities/strings/string_tools.hpp>

#define TEMPLATE_PARAMS template<typename FieldType>
#define TABUFFERPIMPL TABufferPIMPL<FieldType>

/// The various tensor related types
using scal_data_tile_t  = TA::Tensor<double>;
using scal_lazy_tile_t  = tensorwrapper::ta_helpers::LazyTile<scal_data_tile_t>;
using tot_data_tile_t   = TA::Tensor<TA::Tensor<double>>;
using tot_lazy_tile_t   = tensorwrapper::ta_helpers::LazyTile<tot_data_tile_t>;
using scal_data_array_t = TA::DistArray<scal_data_tile_t, TA::SparsePolicy>;
using scal_lazy_array_t = TA::DistArray<scal_lazy_tile_t, TA::SparsePolicy>;
using tot_data_array_t  = TA::DistArray<tot_data_tile_t, TA::SparsePolicy>;
using tot_lazy_array_t  = TA::DistArray<tot_lazy_tile_t, TA::SparsePolicy>;

namespace tensorwrapper::tensor::buffer::detail_ {
namespace {

// TODO: these should be replaced by the Conversion class
template<typename FieldType>
auto& downcast(BufferPIMPL<FieldType>& input) {
    using to_type = TABufferPIMPL<FieldType>;
    auto ptr      = dynamic_cast<to_type*>(&input);
    if(ptr == nullptr) { throw std::bad_cast(); }
    return *ptr;
}

template<typename FieldType>
const auto& downcast(const BufferPIMPL<FieldType>& input) {
    using to_type = TABufferPIMPL<FieldType>;
    auto ptr      = dynamic_cast<const to_type*>(&input);
    if(ptr == nullptr) { throw std::bad_cast(); }
    return *ptr;
}

template<typename Variant0, typename Variant1, typename Fxn>
void double_call(Variant0&& v0, Variant1&& v1, Fxn&& fxn) {
    auto l = [&](auto&& lhs) {
        auto m = [&](auto&& rhs) { fxn(lhs, rhs); };
        std::visit(m, v1);
    };
    std::visit(l, std::forward<Variant0>(v0));
}

template<typename Variant0, typename Variant1, typename Fxn>
auto double_call_w_return(Variant0&& v0, Variant1&& v1, Fxn&& fxn) {
    auto l = [&](auto&& lhs) {
        auto m = [&](auto&& rhs) { return fxn(lhs, rhs); };
        return std::visit(m, v1);
    };
    return std::visit(l, std::forward<Variant0>(v0));
}

template<typename Variant0, typename Variant1, typename Variant2, typename Fxn>
void triple_call(Variant0&& v0, Variant1&& v1, Variant2&& v2, Fxn&& fxn) {
    auto l = [&](auto&& t0) {
        auto m = [&](auto&& t1) {
            auto n = [&](auto&& t2) { return fxn(t0, t1, t2); };
            std::visit(n, v2);
        };
        std::visit(m, v1);
    };
    std::visit(l, std::forward<Variant0>(v0));
}

/// Default pass-through if tile is already a data tile
tot_data_tile_t as_data_tile(const tot_data_tile_t& t) { return t; }

/// Convert lazy tile to data
tot_data_tile_t as_data_tile(tot_lazy_tile_t t) {
    auto t_as_data = tot_data_tile_t(t);
    return std::move(t_as_data);
}

// -- Guts Functions -----------------------------------------------------------
// These exist mostly to deal with data vs lazy array differences and typing.

template<typename OutType>
void retile_guts_(OutType& t, TA::TiledRange trange) {
    if constexpr(std::is_same_v<OutType, scal_data_array_t>) {
        t = TA::retile(t, std::move(trange));
    } else if constexpr(std::is_same_v<OutType, tot_data_array_t>) {
        throw std::runtime_error("retile NYI for ToTs!!!!");
    } else {
        throw std::runtime_error("retile NYI for Lazy Arrays!!!!");
    }
}

template<typename OutType>
void set_shape_guts_(OutType& t, TA::SparseShape<float> new_shape) {
    if constexpr(std::is_same_v<OutType, scal_data_array_t> ||
                 std::is_same_v<OutType, tot_data_array_t>) {
        auto outer_rank                 = t.trange().rank();
        decltype(outer_rank) inner_rank = 0;
        if constexpr(std::is_same_v<OutType, tot_data_array_t>) {
            if(t.is_initialized()) {
                const auto& tile0 = t.begin()->get();
                inner_rank        = as_data_tile(tile0)[0].range().rank();
            }
        }
        auto idx = TA::detail::dummy_annotation(outer_rank, inner_rank);
        t(idx)   = t(idx).set_shape(std::move(new_shape));
    } else {
        throw std::runtime_error("set_shape NYI for Lazy Arrays!!!!");
    }
}

template<typename LHSType, typename RHSType>
bool are_equal_guts_(const LHSType& lhs, const RHSType& rhs) {
    if constexpr(!std::is_same_v<LHSType, RHSType>) {
        return false;
    } else if constexpr(std::is_same_v<LHSType, scal_lazy_array_t> ||
                        std::is_same_v<LHSType, tot_lazy_array_t>) {
        /// TODO: Actually evaluate these somehow
        return false;
    } else {
        return lhs == rhs;
    }
}

template<typename OutType, typename MyType>
void permute_guts_(OutType& out, MyType& rhs, std::string out_idx,
                   std::string rhs_idx) {
    if constexpr(std::is_same_v<OutType, scal_lazy_array_t> ||
                 std::is_same_v<OutType, tot_lazy_array_t>) {
        throw std::runtime_error("Cannot assign to lazy array.");
    } else {
        out(out_idx) = rhs(rhs_idx);
    }
}

template<typename OutType, typename MyType>
void scale_guts_(OutType& out, MyType& lhs, double rhs, std::string out_idx,
                 std::string lhs_idx) {
    if constexpr(std::is_same_v<OutType, scal_lazy_array_t> ||
                 std::is_same_v<OutType, tot_lazy_array_t>) {
        throw std::runtime_error("Cannot assign to lazy array.");
    } else {
        out(out_idx) = lhs(lhs_idx) * rhs;
    }
}

template<typename OutType, typename LHSType, typename RHSType>
void add_guts_(OutType& out, LHSType& lhs, RHSType rhs, std::string out_idx,
               std::string lhs_idx, std::string rhs_idx) {
    if constexpr(std::is_same_v<OutType, scal_lazy_array_t> ||
                 std::is_same_v<OutType, tot_lazy_array_t>) {
        throw std::runtime_error("Cannot assign to lazy array.");
    } else {
        out(out_idx) = lhs(lhs_idx) + rhs(rhs_idx);
    }
}

template<typename OutType, typename RHSType>
void inplace_add_guts_(OutType& out, RHSType rhs, std::string out_idx,
                       std::string rhs_idx) {
    if constexpr(std::is_same_v<OutType, scal_lazy_array_t> ||
                 std::is_same_v<OutType, tot_lazy_array_t>) {
        throw std::runtime_error("Cannot assign to lazy array.");
    } else {
        out(out_idx) += rhs(rhs_idx);
    }
}

template<typename OutType, typename LHSType, typename RHSType>
void subtract_guts_(OutType& out, LHSType& lhs, RHSType rhs,
                    std::string out_idx, std::string lhs_idx,
                    std::string rhs_idx) {
    if constexpr(std::is_same_v<OutType, scal_lazy_array_t> ||
                 std::is_same_v<OutType, tot_lazy_array_t>) {
        throw std::runtime_error("Cannot assign to lazy array.");
    } else {
        out(out_idx) = lhs(lhs_idx) - rhs(rhs_idx);
    }
}

template<typename OutType, typename RHSType>
void inplace_subtract_guts_(OutType& out, RHSType rhs, std::string out_idx,
                            std::string rhs_idx) {
    if constexpr(std::is_same_v<OutType, scal_lazy_array_t> ||
                 std::is_same_v<OutType, tot_lazy_array_t>) {
        throw std::runtime_error("Cannot assign to lazy array.");
    } else {
        out(out_idx) -= rhs(rhs_idx);
    }
}

template<typename OutType, typename LHSType, typename RHSType>
void times_guts_(OutType& out, LHSType& lhs, RHSType rhs, std::string out_idx,
                 std::string lhs_idx, std::string rhs_idx) {
    if constexpr(std::is_same_v<OutType, scal_lazy_array_t> ||
                 std::is_same_v<OutType, tot_lazy_array_t>) {
        throw std::runtime_error("Cannot assign to lazy array.");
    } else {
        out(out_idx) = lhs(lhs_idx) * rhs(rhs_idx);
    }
}

template<typename OutType, typename LHSType, typename RHSType>
void einsum_if_data(const std::string& out_idx, const std::string& lhs_idx,
                    const std::string& rhs_idx, const LHSType& lhs,
                    const RHSType& rhs, OutType& out) {
    if constexpr(std::is_same_v<OutType, TA::TSpArrayD> &&
                 std::is_same_v<LHSType, TA::TSpArrayD> &&
                 std::is_same_v<RHSType, TA::TSpArrayD>) {
        using ta_helpers::einsum::einsum;
        out = einsum(out_idx, lhs_idx, rhs_idx, lhs, rhs);
    } else {
        throw std::runtime_error(
          "Einsum inputs and outputs must be data arrays");
    }
}

} // namespace

// -- Constructors -------------------------------------------------------------

TEMPLATE_PARAMS
TABUFFERPIMPL::TABufferPIMPL(default_tensor_type t2wrap) :
  m_tensor_(std::move(t2wrap)) {}

TEMPLATE_PARAMS
TABUFFERPIMPL::TABufferPIMPL(lazy_tensor_type t2wrap) :
  m_tensor_(std::move(t2wrap)) {}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::pimpl_pointer TABUFFERPIMPL::default_clone_() const {
    return std::unique_ptr<base_type>(new TABufferPIMPL());
}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::pimpl_pointer TABUFFERPIMPL::clone_() const {
    auto clone_pimpl    = default_clone_();
    auto& clone_variant = downcast(*(clone_pimpl.get())).m_tensor_;
    auto l              = [&](auto&& arg) { clone_variant = TA::clone(arg); };
    std::visit(l, m_tensor_);
    return clone_pimpl;
}

// -- Setters ------------------------------------------------------------------

TEMPLATE_PARAMS
void TABUFFERPIMPL::retile(ta_trange_type trange) {
    auto l = [trange{std::move(trange)}](auto&& arg) {
        retile_guts_(arg, trange);
    };
    std::visit(l, m_tensor_);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::set_shape(ta_shape_type new_shape) {
    auto l = [new_shape{std::move(new_shape)}](auto&& t) {
        set_shape_guts_(t, new_shape);
    };
    std::visit(l, m_tensor_);
}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::scalar_value_type TABUFFERPIMPL::norm_() const {
    return std::visit(
      [=](auto&& t) {
          auto outer_rank                 = t.trange().rank();
          decltype(outer_rank) inner_rank = 0;
          if constexpr(std::is_same_v<FieldType, field::Tensor>) {
              if(t.is_initialized()) {
                  const auto& tile0 = t.begin()->get();
                  inner_rank        = as_data_tile(tile0)[0].range().rank();
              }
          }
          auto idx = TA::detail::dummy_annotation(outer_rank, inner_rank);

          return t(idx).norm().get();
      },
      m_tensor_);
}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::scalar_value_type TABUFFERPIMPL::sum_() const {
    return std::visit(
      [=](auto&& t) {
          auto outer_rank                 = t.trange().rank();
          decltype(outer_rank) inner_rank = 0;
          if constexpr(std::is_same_v<FieldType, field::Tensor>) {
              if(t.is_initialized()) {
                  const auto& tile0 = t.begin()->get();
                  inner_rank        = as_data_tile(tile0)[0].range().rank();
              }
          }
          auto idx = TA::detail::dummy_annotation(outer_rank, inner_rank);

          return t(idx).sum().get();
      },
      m_tensor_);
}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::scalar_value_type TABUFFERPIMPL::trace_() const {
    if constexpr(std::is_same_v<FieldType, field::Tensor>) {
        throw std::runtime_error("Trace not implemented for ToT");
        return 0.0;
    } else {
        return std::visit(
          [=](auto&& t) {
              auto trange = t.trange();
              auto rank   = t.trange().rank();
              if(rank != 2 or (trange.dim(0) != trange.dim(1))) {
                  throw std::runtime_error(
                    "Trace not defined for non-square matrix");
                  return 0.0;
              }
              auto idx = TA::detail::dummy_annotation(2, 0);

              return t(idx).trace().get();
          },
          m_tensor_);
    }
}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::extents_type TABUFFERPIMPL::make_extents_() const {
    using size_type = typename extents_type::size_type;

    auto l = [=](auto&& t) {
        if(!t.is_initialized()) return extents_type{};
        const auto& tr = t.trange();
        extents_type rv(tr.rank());
        const auto& erange = tr.elements_range().extent();
        for(size_type i = 0; i < rv.size(); ++i) rv[i] = erange[i];
        return rv;
    };
    return std::visit(l, m_tensor_);
}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::inner_extents_type TABUFFERPIMPL::make_inner_extents_()
  const {
    if constexpr(std::is_same_v<FieldType, field::Scalar>)
        /// ScalarTensorWrapper has no inner extents.
        return 1;
    else if constexpr(std::is_same_v<FieldType, field::Tensor>) {
        /// Shape inner_extents_type = std::map<index_type, Shape<field::Scalar>
        using index_type = typename inner_extents_type::key_type;
        using shape_type = typename inner_extents_type::mapped_type;

        auto l = [=](auto&& t) {
            inner_extents_type rv{};
            if(!t.is_initialized()) return rv;
            for(const auto& tile : t) {
                auto inner_tile = as_data_tile(tile.get());
                for(auto i = 0; i < inner_tile.size(); ++i) {
                    /// Inner tensor info
                    auto idx_i    = inner_tile.range().idx(i);
                    auto& range_i = inner_tile[i].range();

                    /// Make Index and Shape for map
                    index_type idx(idx_i.begin(), idx_i.end());
                    extents_type extents_i(range_i.rank());
                    for(auto i = 0; i < extents_i.size(); ++i)
                        extents_i[i] = range_i.upbound(i);
                    shape_type inner_shape(extents_i);

                    /// Add to map
                    rv[idx] = inner_shape;
                }
            }
            return rv;
        };
        return std::visit(l, m_tensor_);
    }
}

// -- Utilities ----------------------------------------------------------------

TEMPLATE_PARAMS
void TABUFFERPIMPL::hash_(hasher_reference h) const {
    std::visit([&](auto&& t) { h(t); }, m_tensor_);
}

TEMPLATE_PARAMS
bool TABUFFERPIMPL::are_equal_(const base_type& rhs) const noexcept {
    // Attempt to downcast
    auto ptr = dynamic_cast<const my_type*>(&rhs);

    // If it failed rhs is not a TABufferPIMPL instance
    if(ptr == nullptr) return false;

    auto l = [&](auto&& lhs, auto&& rhs) { return are_equal_guts_(lhs, rhs); };
    return double_call_w_return(m_tensor_, ptr->m_tensor_, l);
}

TEMPLATE_PARAMS
std::string TABUFFERPIMPL::to_str_() const {
    auto l = [=](auto&& t) {
        std::stringstream ss;
        ss << t;
        return ss.str();
    };
    return std::visit(l, m_tensor_);
}

// -- Math Ops -----------------------------------------------------------------

TEMPLATE_PARAMS
void TABUFFERPIMPL::permute_(const_annotation_reference my_idx,
                             const_annotation_reference out_idx,
                             base_type& out) const {
    auto& out_tensor = downcast(out).m_tensor_;
    auto l           = [&](auto&& out, auto&& rhs) {
        permute_guts_(out, rhs, out_idx, my_idx);
    };
    double_call(out_tensor, m_tensor_, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::scale_(const_annotation_reference my_idx,
                           const_annotation_reference out_idx, base_type& out,
                           double rhs) const {
    auto& out_tensor = downcast(out).m_tensor_;
    auto l           = [&](auto&& out, auto&& lhs) {
        scale_guts_(out, lhs, rhs, out_idx, my_idx);
    };
    double_call(out_tensor, m_tensor_, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::add_(const_annotation_reference my_idx,
                         const_annotation_reference out_idx, base_type& out,
                         const_annotation_reference rhs_idx,
                         const base_type& rhs) const {
    auto& out_tensor       = downcast(out).m_tensor_;
    const auto& rhs_tensor = downcast(rhs).m_tensor_;
    auto l                 = [&](auto&& out, auto&& lhs, auto&& rhs) {
        add_guts_(out, lhs, rhs, out_idx, my_idx, rhs_idx);
    };
    triple_call(out_tensor, m_tensor_, rhs_tensor, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::inplace_add_(const_annotation_reference my_idx,
                                 const_annotation_reference rhs_idx,
                                 const base_type& rhs) {
    const auto& rhs_tensor = downcast(rhs).m_tensor_;
    auto l                 = [&](auto&& lhs, auto&& rhs) {
        inplace_add_guts_(lhs, rhs, my_idx, rhs_idx);
    };
    double_call(m_tensor_, rhs_tensor, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::subtract_(const_annotation_reference my_idx,
                              const_annotation_reference out_idx,
                              base_type& out,
                              const_annotation_reference rhs_idx,
                              const base_type& rhs) const {
    auto& out_tensor       = downcast(out).m_tensor_;
    const auto& rhs_tensor = downcast(rhs).m_tensor_;
    auto l                 = [&](auto&& out, auto&& lhs, auto&& rhs) {
        subtract_guts_(out, lhs, rhs, out_idx, my_idx, rhs_idx);
    };
    triple_call(out_tensor, m_tensor_, rhs_tensor, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::inplace_subtract_(const_annotation_reference my_idx,
                                      const_annotation_reference rhs_idx,
                                      const base_type& rhs) {
    const auto& rhs_tensor = downcast(rhs).m_tensor_;
    auto l                 = [&](auto&& lhs, auto&& rhs) {
        inplace_subtract_guts_(lhs, rhs, my_idx, rhs_idx);
    };
    double_call(m_tensor_, rhs_tensor, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::times_(const_annotation_reference my_idx,
                           const_annotation_reference out_idx, base_type& out,
                           const_annotation_reference rhs_idx,
                           const base_type& rhs) const {
    /// Grab our tensors
    auto& out_tensor       = downcast(out).m_tensor_;
    const auto& rhs_tensor = downcast(rhs).m_tensor_;
    auto l                 = [&](auto&& out, auto&& lhs, auto&& rhs) {
        times_guts_(out, lhs, rhs, out_idx, my_idx, rhs_idx);
    };

    if constexpr(std::is_same_v<FieldType, field::Scalar>) {
        /// Do we need to use einsum?
        using utilities::strings::split_string;
        const auto& lidx = split_string(my_idx, ",");
        const auto& ridx = split_string(rhs_idx, ",");
        const auto& oidx = split_string(out_idx, ",");
        bool use_einsum  = false;
        for(const auto& x : oidx) {
            const auto l_count = std::count(lidx.begin(), lidx.end(), x);
            const auto r_count = std::count(ridx.begin(), ridx.end(), x);
            if(l_count == 1 && r_count == 1) {
                use_einsum = true;
                break;
            }
        }

        if(use_einsum) {
            /// Einsum lambda
            auto l_einsum = [&](auto&& out, auto&& lhs, auto&& rhs) {
                einsum_if_data(out_idx, my_idx, rhs_idx, lhs, rhs, out);
            };
            triple_call(out_tensor, m_tensor_, rhs_tensor, l_einsum);
        } else {
            triple_call(out_tensor, m_tensor_, rhs_tensor, l);
        }
    } else {
        triple_call(out_tensor, m_tensor_, rhs_tensor, l);
    }
}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::scalar_value_type TABUFFERPIMPL::dot_(
  const_annotation_reference my_idx, const_annotation_reference rhs_idx,
  const base_type& rhs) const {
    const auto& rhs_tensor = downcast(rhs).m_tensor_;

    auto l = [&](auto&& lhs_in, auto&& rhs_in) {
        return lhs_in(my_idx).dot(rhs_in(rhs_idx));
    };
    return double_call_w_return(m_tensor_, rhs_tensor, l);
}

template class TABufferPIMPL<field::Scalar>;
template class TABufferPIMPL<field::Tensor>;

} // namespace tensorwrapper::tensor::buffer::detail_

#undef TEMPLATE_PARAMS
#undef TABUFFERPIMPL
