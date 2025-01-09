#pragma once
#include "shapes.hpp"
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper::testing {

// -----------------------------------------------------------------------------
// -- Logical layouts
// -----------------------------------------------------------------------------

inline auto scalar_logical() {
    return tensorwrapper::layout::Logical(smooth_scalar());
}

inline auto vector_logical(std::size_t i = 10) {
    return tensorwrapper::layout::Logical(smooth_vector(i));
}

inline auto matrix_logical(std::size_t i = 10, std::size_t j = 10) {
    return tensorwrapper::layout::Logical(smooth_matrix(i, j));
}

inline auto tensor_logical(std::size_t i = 10, std::size_t j = 10,
                           std::size_t k = 10) {
    return tensorwrapper::layout::Logical(smooth_tensor(i, j, k));
}

// -----------------------------------------------------------------------------
// -- Physical layouts
// -----------------------------------------------------------------------------

inline auto scalar_physical() {
    return tensorwrapper::layout::Physical(smooth_scalar());
}

inline auto vector_physical(std::size_t i = 10) {
    return tensorwrapper::layout::Physical(smooth_vector(i));
}

inline auto matrix_physical(std::size_t i = 10, std::size_t j = 10) {
    return tensorwrapper::layout::Physical(smooth_matrix(i, j));
}

inline auto tensor_physical(std::size_t i = 10, std::size_t j = 10,
                            std::size_t k = 10) {
    return tensorwrapper::layout::Physical(smooth_tensor(i, j, k));
}

} // namespace tensorwrapper::testing