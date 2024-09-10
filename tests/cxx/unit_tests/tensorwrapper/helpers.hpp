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

namespace tensorwrapper::testing {

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
