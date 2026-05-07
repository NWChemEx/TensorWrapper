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
template<typename T>
using uncertain_type = sigma::Uncertain<T>;
using ufloat         = uncertain_type<float>;
using udouble        = uncertain_type<double>;
template<typename T>
using interval_type = sigma::Interval<T>;
using ifloat        = interval_type<float>;
using idouble       = interval_type<double>;

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
    if constexpr(is_uncertain_v<T>) {
        return sigma::fabs(value);
    } else if constexpr(is_interval_v<T>) {
        return T(sigma::fabs(value));
    } else {
        return std::fabs(value);
    }
}

template<typename T>
T log(T value) {
    if constexpr(is_uncertain_v<T>) {
        return sigma::log(value);
    } else if constexpr(is_interval_v<T>) {
        return T(sigma::log(value));
    } else {
        return std::log(value);
    }
}

template<typename T>
T exp(T value) {
    if constexpr(is_uncertain_v<T>) {
        return sigma::exp(value);
    } else if constexpr(is_interval_v<T>) {
        return T(sigma::exp(value));
    } else {
        return std::exp(value);
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
template<typename T>
using uncertain_type = T;
using ufloat         = uncertain_type<float>;
using udouble        = uncertain_type<double>;
template<typename T>
using interval_type = T;
using ifloat        = float;
using idouble       = double;

using floating_point_types = std::tuple<float, double>;

template<typename>
constexpr bool is_uncertain_v = false;

template<typename T>
constexpr bool is_interval_v = false;

template<typename T>
T fabs(T value) {
    return std::fabs(value);
}

template<typename T>
T log(T value) {
    return std::log(value);
}

template<typename T>
T exp(T value) {
    return std::exp(value);
}

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double)

} // namespace tensorwrapper::types
#endif
