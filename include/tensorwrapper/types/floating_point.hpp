/*
 * Copyright 2025 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

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

template<typename T>
constexpr bool is_uncertain_v =
  std::is_same_v<T, ufloat> || std::is_same_v<T, udouble>;

template<typename T>
T fabs(T value) {
    if constexpr(is_uncertain_v<T>) {
        return sigma::fabs(value);
    } else {
        return std::fabs(value);
    }
}

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double);                           \
    MACRO_IN(types::ufloat);                    \
    MACRO_IN(types::udouble)

#else
using ufloat  = float;
using udouble = double;

using floating_point_types = std::tuple<float, double>;

template<typename>
constexpr bool is_uncertain_v = false;

template<typename T>
T fabs(T value) {
    return std::fabs(value);
}

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double)

#endif

} // namespace tensorwrapper::types
