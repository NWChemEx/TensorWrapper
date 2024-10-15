#pragma once
#include <memory>
#include <tensorwrapper/shape/shape_fwd.hpp>

namespace tensorwrapper::shape {

template<typename ShapeType>
struct ShapeTraits;

template<>
struct ShapeTraits<ShapeBase> {
    using shape_base   = ShapeBase;
    using base_pointer = std::unique_ptr<shape_base>;
    using rank_type    = unsigned short;
    using size_type    = std::size_t;
};

template<>
struct ShapeTraits<const ShapeBase> {
    using shape_base   = ShapeBase;
    using base_pointer = std::unique_ptr<shape_base>;
    using rank_type    = unsigned short;
    using size_type    = std::size_t;
};

template<>
struct ShapeTraits<Smooth> : public ShapeTraits<ShapeBase> {
    using value_type       = Smooth;
    using const_value_type = const value_type;
    using reference        = value_type&;
    using const_reference  = const value_type&;
    using pointer          = value_type*;
    using const_pointer    = const value_type*;
};

template<>
struct ShapeTraits<const Smooth> : public ShapeTraits<const ShapeBase> {
    using value_type       = Smooth;
    using const_value_type = const value_type;
    using reference        = const value_type&;
    using const_reference  = const value_type&;
    using pointer          = const value_type*;
    using const_pointer    = const value_type*;
};

template<typename T>
struct ShapeTraits<SmoothView<T>> {
    using smooth_traits = ShapeTraits<T>;
    using pimpl_type    = detail_::SmoothViewPIMPL<T>;
    using const_pimpl_type =
      detail_::SmoothViewPIMPL<typename smooth_traits::const_value_type>;
    using pimpl_pointer       = std::unique_ptr<pimpl_type>;
    using const_pimpl_pointer = std::unique_ptr<const_pimpl_type>;
};

} // namespace tensorwrapper::shape