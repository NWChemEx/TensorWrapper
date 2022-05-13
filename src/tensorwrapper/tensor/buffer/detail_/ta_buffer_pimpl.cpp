#include "ta_buffer_pimpl.hpp"

#include "tensorwrapper/ta_helpers/ta_helpers.hpp"

#define TEMPLATE_PARAMS template<typename FieldType>
#define TABUFFERPIMPL TABufferPIMPL<FieldType>

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

} // namespace

// -- Constructors -------------------------------------------------------------

TEMPLATE_PARAMS
TABUFFERPIMPL::TABufferPIMPL(default_tensor_type t2wrap) :
  m_tensor_(std::move(t2wrap)) {}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::pimpl_pointer TABUFFERPIMPL::default_clone_() const {
    return std::unique_ptr<base_type>(new TABufferPIMPL());
}

TEMPLATE_PARAMS
typename TABUFFERPIMPL::pimpl_pointer TABUFFERPIMPL::clone_() const {
    // Can't use make_unique b/c copy ctor is protected
    return std::unique_ptr<base_type>(new TABufferPIMPL(*this));
}

// -- Setters ------------------------------------------------------------------

TEMPLATE_PARAMS
void TABUFFERPIMPL::retile(ta_trange_type trange) {
    if constexpr(std::is_same_v<FieldType, field::Scalar>) {
        auto l = [trange{std::move(trange)}](auto&& arg) {
            arg = TA::retile(arg, std::move(trange));
        };
        std::visit(l, m_tensor_);
    } else {
        throw std::runtime_error("retile NYI for ToTs!!!!");
    }
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::set_shape(ta_shape_type new_shape) {
    auto l = [new_shape{std::move(new_shape)}](auto&& t) {
        auto outer_rank                 = t.trange().rank();
        decltype(outer_rank) inner_rank = 0;
        if constexpr(std::is_same_v<FieldType, field::Tensor>) {
            if(t.is_initialized()) {
                const auto& tile0 = t.begin()->get();
                inner_rank        = tile0[0].range().rank();
            }
        }
        auto idx = TA::detail::dummy_annotation(outer_rank, inner_rank);
        t(idx)   = t(idx).set_shape(std::move(new_shape));
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
                  inner_rank        = tile0[0].range().rank();
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
                  inner_rank        = tile0[0].range().rank();
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

    auto l = [&](auto&& lhs, auto&& rhs) { return lhs == rhs; };
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
void TABUFFERPIMPL::scale_(const_annotation_reference my_idx,
                           const_annotation_reference out_idx, base_type& out,
                           double rhs) const {
    auto& out_tensor = downcast(out).m_tensor_;
    auto l = [&](auto&& out, auto&& lhs) { out(out_idx) = lhs(my_idx) * rhs; };
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
        out(out_idx) = lhs(my_idx) + rhs(rhs_idx);
    };
    triple_call(out_tensor, m_tensor_, rhs_tensor, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::inplace_add_(const_annotation_reference my_idx,
                                 const_annotation_reference rhs_idx,
                                 const base_type& rhs) {
    const auto& rhs_tensor = downcast(rhs).m_tensor_;
    auto l = [&](auto&& lhs, auto&& rhs) { lhs(my_idx) += rhs(rhs_idx); };
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
        out(out_idx) = lhs(my_idx) - rhs(rhs_idx);
    };
    triple_call(out_tensor, m_tensor_, rhs_tensor, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::inplace_subtract_(const_annotation_reference my_idx,
                                      const_annotation_reference rhs_idx,
                                      const base_type& rhs) {
    const auto& rhs_tensor = downcast(rhs).m_tensor_;
    auto l = [&](auto&& lhs, auto&& rhs) { lhs(my_idx) -= rhs(rhs_idx); };
    double_call(m_tensor_, rhs_tensor, l);
}

TEMPLATE_PARAMS
void TABUFFERPIMPL::times_(const_annotation_reference my_idx,
                           const_annotation_reference out_idx, base_type& out,
                           const_annotation_reference rhs_idx,
                           const base_type& rhs) const {
    auto& out_tensor       = downcast(out).m_tensor_;
    const auto& rhs_tensor = downcast(rhs).m_tensor_;
    auto l                 = [&](auto&& out, auto&& lhs, auto&& rhs) {
        out(out_idx) = lhs(my_idx) * rhs(rhs_idx);
    };
    triple_call(out_tensor, m_tensor_, rhs_tensor, l);
}

template class TABufferPIMPL<field::Scalar>;
template class TABufferPIMPL<field::Tensor>;

} // namespace tensorwrapper::tensor::buffer::detail_

#undef TEMPLATE_PARAMS
#undef TABUFFERPIMPL
