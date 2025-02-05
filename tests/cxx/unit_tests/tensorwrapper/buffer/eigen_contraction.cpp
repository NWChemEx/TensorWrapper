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

#include "../testing/testing.hpp"
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/buffer/eigen_contraction.hpp>

using namespace tensorwrapper;
using namespace buffer;

#ifdef ENABLE_SIGMA
using types2test = std::tuple<float, double, sigma::UFloat, sigma::UDouble>;
#else
using types2test = std::tuple<float, double>;
#endif

TEMPLATE_LIST_TEST_CASE("eigen_contraction", "", types2test) {
    using float_t    = TestType;
    using mode_type  = unsigned short;
    using pair_type  = std::pair<mode_type, mode_type>;
    using mode_array = std::vector<pair_type>;

    // Inputs
    auto scalar  = testing::eigen_scalar<float_t>();
    auto vector  = testing::eigen_vector<float_t>();
    auto vector2 = testing::eigen_vector<float_t>(2);
    auto matrix  = testing::eigen_matrix<float_t>();

    mode_array m00{pair_type{0, 0}};
    mode_array m11{pair_type{1, 1}};
    mode_array m00_11{pair_type{0, 0}, pair_type{1, 1}};

    auto scalar_corr      = testing::eigen_scalar<float_t>();
    scalar_corr.value()() = 30.0;

    auto vector_corr       = testing::eigen_vector<float_t>(2);
    vector_corr.value()(0) = 3.0;
    vector_corr.value()(1) = 4.0;

    auto matrix_corr          = testing::eigen_matrix<float_t>(2, 2);
    matrix_corr.value()(0, 0) = 10.0;
    matrix_corr.value()(0, 1) = 14.0;
    matrix_corr.value()(1, 0) = 14.0;
    matrix_corr.value()(1, 1) = 20.0;

    SECTION("vector with vector") {
        auto& rv = eigen_contraction<float_t>(scalar, vector, vector, m00);
        REQUIRE(&rv == static_cast<BufferBase*>(&scalar));
        REQUIRE(scalar_corr.are_equal(scalar));
    }

    SECTION("ij,ij->") {
        auto& rv = eigen_contraction<float_t>(scalar, matrix, matrix, m00_11);
        REQUIRE(&rv == static_cast<BufferBase*>(&scalar));
        REQUIRE(scalar_corr.are_equal(scalar));
    }

    // SECTION("ki,kj->ij") {
    //     auto buffer = testing::eigen_matrix<float_t>();
    //     auto& rv    = eigen_contraction<float_t>(buffer, matrix, matrix, m00);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&buffer));
    //     REQUIRE(matrix_corr.are_equal(buffer));
    // }

    // SECTION("ij,i->j") {
    //     auto buffer = testing::eigen_vector<float_t>(2);
    //     auto& rv    = eigen_contraction<float_t>(buffer, matrix, vector2, m00);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&buffer));
    //     REQUIRE(vector_corr.are_equal(rv));
    // }
}