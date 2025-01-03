/** @file shapes.hpp
 *
 *  This file contains some already made shape objects to facilitate unit
 *  testing of TensorWrapper.
 */
#pragma once
#include <tensorwrapper/tensorwrapper.hpp>

namespace test_tensorwrapper {

inline auto smooth_scalar() { return tensorwrapper::shape::Smooth{}; }

inline auto smooth_vector(std::size_t i = 10) {
    return tensorwrapper::shape::Smooth{i};
}

inline auto smooth_matrix(std::size_t i = 10, std::size_t j = 10) {
    return tensorwrapper::shape::Smooth{i, j};
}

inline auto smooth_tensor(std::size_t i = 10, std::size_t j = 10,
                          std::size_t k = 10) {
    return tensorwrapper::shape::Smooth{i, j, k};
}

} // namespace test_tensorwrapper