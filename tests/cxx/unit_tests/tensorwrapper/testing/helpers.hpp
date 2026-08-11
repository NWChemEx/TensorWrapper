/*
 * Copyright 2024 NWChemEx-Project
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
#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <wtf/fp/float_view.hpp>

namespace tensorwrapper::testing {

// Detect FloatView<FloatType> (variable template specialization).
// FloatBuffer::at() / Contiguous::get_elem() return FloatView<Float> where
// Float is the type-erased wtf::fp::Float, not the concrete element type.
// Calling value<FloatType>() on such a view fails because wtf::fp::Float
// itself is not registered as a FloatingPoint type.  Instead, we cast to the
// concrete type given by the OTHER argument (U) in elements_equal.
template<typename T>
inline constexpr bool is_float_view_v = false;
template<typename FloatType>
inline constexpr bool is_float_view_v<wtf::fp::FloatView<FloatType>> = true;

// Compare two buffer elements.  For affine / thresholded-affine types
// (detected from the rhs concrete type after stripping cv-qualifiers) compare
// interval ranges rather than error-symbol maps, which differ across
// independent computations of the same mathematical value.
// When lhs is a FloatView, the concrete value is extracted via float_cast
// using the rhs type as the target (all sigma UQ types are registered with
// WTF_REGISTER_FP_TYPE so they satisfy concepts::FloatingPoint).
template<typename T, typename U>
bool elements_equal(const T& lhs, const U& rhs) {
    using concrete_t = std::remove_cv_t<U>;
    concrete_t lv    = [&]() -> concrete_t {
        if constexpr(is_float_view_v<std::remove_cv_t<T>>) {
            return wtf::fp::float_cast<concrete_t>(lhs);
        } else {
            return lhs;
        }
    }();
    if constexpr(types::is_affine_v<concrete_t> ||
                 types::is_thresholded_affine_v<concrete_t> ||
                 types::is_taylor_model_v<concrete_t>) {
        return lv.range() == rhs.range();
    } else {
        return lv == rhs;
    }
}

template<typename T>
constexpr double default_tolerance() {
    if constexpr(types::is_uq_type_v<T>) {
        // pow() is implemented as exp(log(x)*n); float-precision accumulates
        // ~1e-3 absolute error for values like 42^2 = 1764.
        return 1e-3;
    } else {
        return 1e-16;
    }
}

/// Tests copy ctor assuming operator== works
template<typename T>
void test_copy_ctor(T&& input) {
    // The actual copy
    std::decay_t<T> other(input);
    REQUIRE(other == input);
}

/// Tests move ctor assuming copy ctor and operator== work
template<typename T>
void test_move_ctor(T&& input) {
    std::decay_t<T> corr(input);
    std::decay_t<T> moved(std::move(input));
    REQUIRE(moved == corr);
}

/** @brief Check copy and move ctors for a series of inputs.
 *
 * Convenience function for applying both test_copy_ctor and test_move_ctor to
 * a series of parameters.
 */
template<typename... Args>
void test_copy_and_move_ctors(Args&&... args) {
    SECTION("Copy ctor") { (test_copy_ctor(args), ...); }
    SECTION("Move ctor") { (test_move_ctor(args), ...); }
}

/** @brief Tests copy assignment assuming operator== works
 *
 *  @param[in] input The object to copy.
 *  @param[in] empty An object to copy @p input in to. If not provided, @p empty
 *                   will be initialized with an empty initializer list.
 */
template<typename T, typename U = std::decay_t<T>>
void test_copy_assignment(T&& input, U&& empty = std::decay_t<T>{}) {
    auto pempty = &(empty = input);
    REQUIRE(empty == input);
    REQUIRE(pempty == &empty);
}

/** @brief Tests move assignment assuming copy ctor and operator== work
 *
 *  @param[in] input The object to move.
 *  @param[in] empty An object to move @p input in to. If not provided, @p empty
 *                   will be initialized with an empty initializer list.
 */
template<typename T, typename U = std::decay_t<T>>
void test_move_assignment(T&& input, U&& empty = std::decay_t<T>{}) {
    std::decay_t<T> corr(input);
    auto pempty = &(empty = std::move(input));
    REQUIRE(empty == corr);
    REQUIRE(pempty == &empty);
}

/** @brief Tests copy and move ctors and assignment operators on a series of
 *         parameters.
 *
 *  This method only works if the default initialization for
 *  test_copy_assignment and test_move_assignment is acceptable.
 */
template<typename... Args>
void test_copy_move_ctor_and_assignment(Args&&... args) {
    test_copy_and_move_ctors(args...);
    SECTION("Copy assignment") { (test_copy_assignment(args), ...); }
    SECTION("Move assignment") { (test_move_assignment(args), ...); }
}

} // namespace tensorwrapper::testing
