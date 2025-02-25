#pragma once
#include <cstddef>

namespace tensorwrapper::types {

template<typename FloatType, std::size_t Rank>
struct ILTraits;

template<typename FloatType>
struct ILTraits<FloatType, 0> {
    using type = FloatType;
};

template<typename FloatType, std::size_t Rank>
struct ILTraits {
    using type =
      std::initializer_list<typename ILTraits<FloatType, Rank - 1>::type>;
};

} // namespace tensorwrapper::types