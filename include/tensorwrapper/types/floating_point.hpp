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
#include <cmath>
#include <tuple>
#include <wtf/wtf.hpp>
#ifdef ENABLE_SIGMA
#include <sigma/sigma.hpp>
#endif

namespace tensorwrapper::types {

#ifdef ENABLE_SIGMA
using ufloat  = sigma::UFloat;
using udouble = sigma::UDouble;
template<typename T>
using interval_type = sigma::GeneralInterval<T>;
using ifloat        = sigma::GeneralInterval<sigma::IFloat>;
using idouble       = sigma::GeneralInterval<sigma::IDouble>;

using floating_point_types =
  std::tuple<float, double, ufloat, udouble, ifloat, idouble>;

template<typename T>
constexpr bool is_uncertain_v =
  std::is_same_v<T, ufloat> || std::is_same_v<T, udouble>;

template<typename T>
constexpr bool is_interval_v =
  std::is_same_v<T, ifloat> || std::is_same_v<T, idouble>;

template<typename T>
T fabs(T value) {
    if constexpr(is_uncertain_v<T> || is_interval_v<T>) {
        return sigma::fabs(value);
    } else {
        return std::fabs(value);
    }
}

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double);                           \
    MACRO_IN(types::ufloat);                    \
    MACRO_IN(types::udouble);                   \
    MACRO_IN(types::ifloat);                    \
    MACRO_IN(types::idouble);
} // namespace tensorwrapper::types

WTF_REGISTER_FP_TYPE(tensorwrapper::types::ufloat);
WTF_REGISTER_FP_TYPE(tensorwrapper::types::udouble);
WTF_REGISTER_FP_TYPE(tensorwrapper::types::ifloat);
WTF_REGISTER_FP_TYPE(tensorwrapper::types::idouble);

#else
using ufloat  = float;
using udouble = double;
template<typename T>
using interval_type = T;
using ifloat        = float;
using idouble       = double;

using floating_point_types = std::tuple<float, double, ifloat, idouble>;

template<typename>
constexpr bool is_uncertain_v = false;

template<typename T>
constexpr bool is_interval_v = false;

template<typename T>
T fabs(T value) {
    return std::fabs(value);
}

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double)

} // namespace tensorwrapper::types
#endif
