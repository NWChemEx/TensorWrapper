#pragma once
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper::testing {

inline auto default_input() { return detail_::TensorInput{}; }

inline auto smooth_scalar() {
    shape::Smooth shape{};
    return detail_::TensorInput(shape);
}

inline auto smooth_vector() {
    shape::Smooth shape{5};
    return detail_::TensorInput(shape);
}

inline auto smooth_symmetric_matrix() {
    shape::Smooth shape{3, 3};
    symmetry::Permutation p01{0, 1};
    symmetry::Group g(p01);
    return detail_::TensorInput(shape, g);
}

} // namespace tensorwrapper::testing
