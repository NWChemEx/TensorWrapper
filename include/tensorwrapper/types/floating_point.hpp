#pragma once
#include <tuple>
#ifdef ENABLE_SIGMA
#include <sigma/sigma.hpp>
#endif

namespace tensorwrapper::types {

#ifdef ENABLE_SIGMA
using ufloat  = sigma::UFloat;
using udouble = sigma::UDouble;

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double);                           \
    MACRO_IN(ufloat);                           \
    MACRO_IN(udouble)

#else
using ufloat  = float;
using udouble = double;

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double)

#endif

using floating_point_types = std::tuple<float, double, ufloat, udouble>;

} // namespace tensorwrapper::types