#pragma once
#include <tuple>
#ifdef ENABLE_SIGMA
#include <sigma/sigma.hpp>
#endif

namespace tensorwrapper::types {

#ifdef ENABLE_SIGMA
using ufloat  = sigma::UFloat;
using udouble = sigma::UDouble;

using floating_point_types = std::tuple<float, double, ufloat, udouble>;

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double);                           \
    MACRO_IN(types::ufloat);                    \
    MACRO_IN(types::udouble)

#else
using ufloat  = float;
using udouble = double;

using floating_point_types = std::tuple<float, double>;

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double)

#endif

} // namespace tensorwrapper::types