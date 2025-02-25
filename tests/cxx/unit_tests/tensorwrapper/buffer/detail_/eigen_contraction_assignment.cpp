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

#include "../../testing/testing.hpp"

using namespace tensorwrapper;
using namespace buffer;

using types2test = types::floating_point_types;

TEMPLATE_LIST_TEST_CASE("eigen_contraction_assignment", "", types2test) {
    // using float_t    = TestType;
    // using label_type = typename BufferBase::label_type;

    // // Inputs
    // auto scalar  = testing::eigen_scalar<float_t>();
    // auto vector  = testing::eigen_vector<float_t>();
    // auto vector2 = testing::eigen_vector<float_t>(2);
    // auto matrix  = testing::eigen_matrix<float_t>();

    // auto scalar_corr      = testing::eigen_scalar<float_t>();
    // scalar_corr.value()() = 30.0;

    // auto vector_corr       = testing::eigen_vector<float_t>(2);
    // vector_corr.value()(0) = 3.0;
    // vector_corr.value()(1) = 4.0;

    // auto matrix_corr          = testing::eigen_matrix<float_t>(2, 2);
    // matrix_corr.value()(0, 0) = 10.0;
    // matrix_corr.value()(0, 1) = 14.0;
    // matrix_corr.value()(1, 0) = 14.0;
    // matrix_corr.value()(1, 1) = 20.0;

    // label_type l("");
    // label_type j("j");
    // label_type ij("i,j");

    // auto mij = matrix("i,j");
    // SECTION("vector with vector") {
    //     auto vi  = vector("i");
    //     auto& rv = eigen_contraction(scalar, l, vi, vi);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&scalar));
    //     REQUIRE(scalar_corr.are_equal(scalar));
    // }

    // SECTION("ij,ij->") {
    //     auto& rv = eigen_contraction(scalar, l, mij, mij);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&scalar));
    //     REQUIRE(scalar_corr.are_equal(scalar));
    // }

    // SECTION("ki,kj->ij") {
    //     auto mki    = matrix("k,i");
    //     auto mkj    = matrix("k,j");
    //     auto buffer = testing::eigen_matrix<float_t>();
    //     auto& rv    = eigen_contraction(buffer, ij, mki, mkj);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&buffer));
    //     REQUIRE(matrix_corr.are_equal(buffer));
    // }

    // SECTION("ij,i->j") {
    //     auto vi     = vector2("i");
    //     auto buffer = testing::eigen_vector<float_t>(2);
    //     auto& rv    = eigen_contraction(buffer, j, mij, vi);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&buffer));
    //     REQUIRE(vector_corr.are_equal(rv));
    // }

    // SECTION("ki,jki->j") {
    //     auto tensor     = testing::eigen_tensor3<float_t>(2);
    //     auto matrix2    = testing::eigen_matrix<float_t>(2);
    //     auto buffer     = testing::eigen_vector<float_t>(2);
    //     auto corr       = testing::eigen_vector<float_t>(2);
    //     corr.value()(0) = 30;
    //     corr.value()(1) = 70;

    //     auto tjki = tensor("j,k,i");
    //     auto mki  = matrix2("k,i");
    //     auto& rv  = eigen_contraction(buffer, j, mki, tjki);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&buffer));
    //     REQUIRE(corr.are_equal(rv));
    // }

    // SECTION("ki,jkl->jil") {
    //     auto tensor           = testing::eigen_tensor3<float_t>(2);
    //     auto matrix2          = testing::eigen_matrix<float_t>(2);
    //     auto buffer           = testing::eigen_tensor3<float_t>(2);
    //     auto corr             = testing::eigen_tensor3<float_t>();
    //     corr.value()(0, 0, 0) = 10;
    //     corr.value()(0, 0, 1) = 14;
    //     corr.value()(0, 1, 0) = 14;
    //     corr.value()(0, 1, 1) = 20;

    //     corr.value()(1, 0, 0) = 26;
    //     corr.value()(1, 0, 1) = 30;
    //     corr.value()(1, 1, 0) = 38;
    //     corr.value()(1, 1, 1) = 44;

    //     auto tjki = tensor("j,k,l");
    //     auto mki  = matrix2("k,i");
    //     label_type jil("j,i,l");
    //     auto& rv = eigen_contraction(buffer, jil, mki, tjki);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&buffer));
    //     REQUIRE(corr.are_equal(rv));
    // }

    // SECTION("kl,ijkl->ij") {
    //     auto tensor        = testing::eigen_tensor4<float_t>();
    //     auto matrix2       = testing::eigen_matrix<float_t>(2);
    //     auto buffer        = testing::eigen_matrix<float_t>(2);
    //     auto corr          = testing::eigen_matrix<float_t>();
    //     corr.value()(0, 0) = 30;
    //     corr.value()(0, 1) = 70;
    //     corr.value()(1, 0) = 110;
    //     corr.value()(1, 1) = 150;

    //     auto lt = tensor("i,j,k,l");
    //     auto lm = matrix2("k,l");
    //     label_type jil("i,j");
    //     auto& rv = eigen_contraction(buffer, ij, lm, lt);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&buffer));
    //     REQUIRE(corr.are_equal(rv));
    // }

    // SECTION("kl,ilkj->ij") {
    //     auto tensor        = testing::eigen_tensor4<float_t>();
    //     auto matrix2       = testing::eigen_matrix<float_t>(2);
    //     auto buffer        = testing::eigen_matrix<float_t>(2);
    //     auto corr          = testing::eigen_matrix<float_t>();
    //     corr.value()(0, 0) = 48;
    //     corr.value()(0, 1) = 58;
    //     corr.value()(1, 0) = 128;
    //     corr.value()(1, 1) = 138;

    //     auto lt = tensor("i,l,k,j");
    //     auto lm = matrix2("k,l");
    //     label_type jil("i,j");
    //     auto& rv = eigen_contraction(buffer, ij, lm, lt);
    //     REQUIRE(&rv == static_cast<BufferBase*>(&buffer));
    //     REQUIRE(corr.are_equal(rv));
    // }
}

// SECTION("contraction_") {
//     auto vi  = vector("i");
//     auto mij = matrix("i,j");
//     auto mik = matrix("i,k");
//     auto mjk = matrix("j,k");

//     SECTION("vector with vector") {
//         auto p = &(scalar.multiplication_assignment("", vi, vi));

//         auto scalar_corr      = testing::eigen_scalar<TestType>();
//         scalar_corr.value()() = 500.0; // 10*10 + 20*20
//         REQUIRE(p == &scalar);
//         compare_eigen<TestType>(scalar_corr.value(), scalar.value());
//     }

//     SECTION("ij,ij->") {
//         auto p = &(scalar.multiplication_assignment("", mij, mij));

//         auto scalar_corr      = testing::eigen_scalar<TestType>();
//         scalar_corr.value()() = 9100.0; // 1400 + 7700
//         REQUIRE(p == &scalar);
//         compare_eigen<TestType>(scalar_corr.value(), scalar.value());
//     }

//     SECTION("ki,kj->ij") {
//         auto buffer2 = testing::eigen_matrix<TestType>();
//         auto p = &(buffer2.multiplication_assignment("i,j", mik, mjk));

//         auto matrix_corr          = testing::eigen_matrix<TestType>(2, 2);
//         matrix_corr.value()(0, 0) = 1400.0; // 100 + 400 + 900
//         matrix_corr.value()(0, 1) = 3200.0; // 400 + 1000 + 1800
//         matrix_corr.value()(1, 0) = 3200.0; // 400 + 1000 + 1800
//         matrix_corr.value()(1, 1) = 7700.0; // 1600 + 2500 + 3600

//         REQUIRE(p == &buffer2);
//         compare_eigen<TestType>(matrix_corr.value(), buffer2.value());
//     }

//     SECTION("ij,i->j") {
//         auto buffer1 = testing::eigen_vector<TestType>();
//         auto p       = &(buffer1.multiplication_assignment("j", mij, vi));

//         auto vector_corr       = testing::eigen_vector<TestType>(3);
//         vector_corr.value()(0) = 900.0;  // 10(10) + 20(40)
//         vector_corr.value()(1) = 1200.0; // 10(20) + 20(50)
//         vector_corr.value()(2) = 1500.0; // 10(30) + 20(60)
//         REQUIRE(p == &buffer1);
//         compare_eigen<TestType>(vector_corr.value(), buffer1.value());
//     }
// }