#pragma once
#include "layouts.hpp"
#include "shapes.hpp"
#include <tensorwrapper/tensorwrapper.hpp>

namespace tensorwrapper::testing {

/// Types that participate in the DSL
using dsl_types =
  std::tuple<tensorwrapper::shape::Smooth, tensorwrapper::symmetry::Group,
             tensorwrapper::sparsity::Pattern, tensorwrapper::layout::Logical,
             tensorwrapper::layout::Physical>;

inline auto scalar_values() {
    return dsl_types{smooth_scalar(), tensorwrapper::symmetry::Group(0),
                     tensorwrapper::sparsity::Pattern(0), scalar_logical(),
                     scalar_physical()};
}

inline auto vector_values() {
    return dsl_types{smooth_vector(), tensorwrapper::symmetry::Group(1),
                     tensorwrapper::sparsity::Pattern(1), vector_logical(),
                     vector_physical()};
}

inline auto matrix_values() {
    return dsl_types{smooth_matrix(), tensorwrapper::symmetry::Group(2),
                     tensorwrapper::sparsity::Pattern(2), matrix_logical(),
                     matrix_physical()};
}

} // namespace tensorwrapper::testing