#pragma once
#include "tensorwrapper/sparse_map/sparse_map.hpp"
#include <catch2/catch.hpp>

namespace testing {

template<std::size_t N>
auto make_indices() {
    if constexpr(N == 1) {
        tensorwrapper::sparse_map::Index i0{0}, i1{1}, i2{2}, i3{3}, i4{4};
        return std::make_tuple(i0, i1, i2, i3, i4);
    } else if constexpr(N == 2) {
        tensorwrapper::sparse_map::Index i00{0, 0}, i01{0, 1}, i10{1, 0},
          i11{1, 1};
        return std::make_tuple(i00, i01, i10, i11);
    } else {
        static_assert(N == 1, "Only support rank 1 and 2 indices");
    }
}

} // namespace testing
