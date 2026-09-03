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

template<typename T>
using affine_type = sigma::Affine<T>;
using afloat      = affine_type<float>;
using adouble     = affine_type<double>;

template<typename T>
using thresholded_affine_type = sigma::ThresholdedAffine<T>;
using tafloat                 = thresholded_affine_type<float>;
using tadouble                = thresholded_affine_type<double>;

template<typename T>
using taylor_model_type = sigma::TaylorModel<T>;
using tmfloat           = taylor_model_type<float>;
using tmdouble          = taylor_model_type<double>;

using floating_point_types =
  std::tuple<float, double, ufloat, udouble, ifloat, idouble, afloat, adouble,
             tafloat, tadouble, tmfloat, tmdouble>;

template<typename T>
constexpr bool is_uncertain_v =
  std::is_same_v<T, ufloat> || std::is_same_v<T, udouble>;

template<typename T>
constexpr bool is_interval_v =
  std::is_same_v<T, ifloat> || std::is_same_v<T, idouble>;

template<typename T>
constexpr bool is_affine_v =
  std::is_same_v<T, afloat> || std::is_same_v<T, adouble>;

template<typename T>
constexpr bool is_thresholded_affine_v =
  std::is_same_v<T, tafloat> || std::is_same_v<T, tadouble>;

template<typename T>
constexpr bool is_taylor_model_v =
  std::is_same_v<T, tmfloat> || std::is_same_v<T, tmdouble>;

template<typename T>
constexpr bool is_uq_type_v =
  is_uncertain_v<T> || is_interval_v<T> || is_affine_v<T> ||
  is_thresholded_affine_v<T> || is_taylor_model_v<T>;

template<typename T, typename U, typename V>
T construct_uq_type(const U& center, const V& radius) {
    if constexpr(is_uncertain_v<T>) {
        return T(center, radius);
    } else if constexpr(is_interval_v<T>) {
        return T(center - radius, center + radius);
    } else if constexpr(is_affine_v<T> || is_thresholded_affine_v<T> ||
                        is_taylor_model_v<T>) {
        return T(center - radius, center + radius);
    } else if constexpr(is_uq_type_v<T>) {
        throw std::logic_error("UQ type not recognized in construct_uq_type.");
    } else {
        return T(center + radius);
    }
}

template<typename T>
auto uq_center(const T& value) {
    if constexpr(is_uncertain_v<T>) {
        return value.mean();
    } else if constexpr(is_interval_v<T>) {
        return value.median();
    } else if constexpr(is_affine_v<T> || is_thresholded_affine_v<T>) {
        return value.center();
    } else if constexpr(is_taylor_model_v<T>) {
        return value.constant();
    } else if constexpr(is_uq_type_v<T>) {
        throw std::logic_error("UQ type not recognized in uq_center.");
    } else {
        return value;
    }
}

template<typename T>
auto uq_upper(const T& value) {
    if constexpr(is_uncertain_v<T>) {
        return value.mean() + value.sd();
    } else if constexpr(is_interval_v<T>) {
        return value.upper();
    } else if constexpr(is_affine_v<T> || is_thresholded_affine_v<T> ||
                        is_taylor_model_v<T>) {
        return value.range().upper();
    } else if constexpr(is_uq_type_v<T>) {
        throw std::logic_error("UQ type not recognized in uq_upper.");
    } else {
        return value;
    }
}

template<typename T>
auto uq_lower(const T& value) {
    if constexpr(is_uncertain_v<T>) {
        return value.mean() - value.sd();
    } else if constexpr(is_interval_v<T>) {
        return value.lower();
    } else if constexpr(is_affine_v<T> || is_thresholded_affine_v<T> ||
                        is_taylor_model_v<T>) {
        return value.range().lower();
    } else if constexpr(is_uq_type_v<T>) {
        throw std::logic_error("UQ type not recognized in uq_lower.");
    } else {
        return value;
    }
}

template<typename T, typename U>
bool strictly_less(const T& lhs, const U& rhs) {
    return uq_upper(lhs) < uq_upper(rhs);
}

template<typename T>
T fabs(T value) {
    if constexpr(is_uq_type_v<T>) {
        return static_cast<T>(sigma::fabs(value));
    } else {
        return std::fabs(value);
    }
}

template<typename T>
T log(T value) {
    if constexpr(is_uq_type_v<T>) {
        return static_cast<T>(sigma::log(value));
    } else {
        return std::log(value);
    }
}

template<typename T>
T exp(T value) {
    if constexpr(is_uq_type_v<T>) {
        return static_cast<T>(sigma::exp(value));
    } else {
        return std::exp(value);
    }
}

template<typename T>
T pow(T value, double pow) {
    if constexpr(is_uq_type_v<T>) {
        return static_cast<T>(sigma::pow(value, pow));
    } else {
        return std::pow(value, pow);
    }
}

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double);                           \
    MACRO_IN(tensorwrapper::types::ufloat);     \
    MACRO_IN(tensorwrapper::types::udouble);    \
    MACRO_IN(tensorwrapper::types::ifloat);     \
    MACRO_IN(tensorwrapper::types::idouble);    \
    MACRO_IN(tensorwrapper::types::afloat);     \
    MACRO_IN(tensorwrapper::types::adouble);    \
    MACRO_IN(tensorwrapper::types::tafloat);    \
    MACRO_IN(tensorwrapper::types::tadouble);   \
    MACRO_IN(tensorwrapper::types::tmfloat);    \
    MACRO_IN(tensorwrapper::types::tmdouble)
} // namespace tensorwrapper::types

WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::ufloat, "ufloat");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::udouble, "udouble");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::ifloat, "ifloat");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::idouble, "idouble");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::afloat, "afloat");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::adouble, "adouble");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::tafloat, "tafloat");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::tadouble, "tadouble");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::tmfloat, "tmfloat");
WTF_REGISTER_FP_TYPE_AS(tensorwrapper::types::tmdouble, "tmdouble");

#else
template<typename T>
using uncertain_type = T;
using ufloat         = uncertain_type<float>;
using udouble        = uncertain_type<double>;
template<typename T>
using interval_type = T;
using ifloat        = float;
using idouble       = double;
template<typename T>
using affine_type = T;
using afloat      = float;
using adouble     = double;
template<typename T>
using thresholded_affine_type = T;
using tafloat                 = float;
using tadouble                = double;
template<typename T>
using taylor_model_type = T;
using tmfloat           = float;
using tmdouble          = double;

using floating_point_types = std::tuple<float, double>;

template<typename>
constexpr bool is_uncertain_v = false;

template<typename T>
constexpr bool is_interval_v = false;

template<typename T>
constexpr bool is_affine_v = false;

template<typename T>
constexpr bool is_thresholded_affine_v = false;

template<typename T>
constexpr bool is_taylor_model_v = false;

template<typename T>
constexpr bool is_uq_type_v = false;

template<typename T, typename U, typename V>
T construct_uq_type(const U& center, const V&) {
    return T(center);
}

template<typename T>
T uq_center(const T& value) {
    return value;
}

template<typename T>
T uq_upper(const T& value) {
    return value;
}

template<typename T>
T uq_lower(const T& value) {
    return value;
}

template<typename T, typename U>
bool strictly_less(const T& lhs, const U& rhs) {
    return lhs < rhs;
    ;
}

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

template<typename T>
T pow(T value, double pow) {
    return std::pow(value, pow);
}

#define TW_APPLY_FLOATING_POINT_TYPES(MACRO_IN) \
    MACRO_IN(float);                            \
    MACRO_IN(double)

} // namespace tensorwrapper::types
#endif

#define DECLARE_WTF_CONTIGUOUS(TYPE)                                   \
    extern template class wtf::buffer::detail_::ContiguousModel<TYPE>; \
    extern template class wtf::buffer::detail_::ContiguousViewModel<TYPE>;

TW_APPLY_FLOATING_POINT_TYPES(DECLARE_WTF_CONTIGUOUS);

#undef DECLARE_WTF_CONTIGUOUS
