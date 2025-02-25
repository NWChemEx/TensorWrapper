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

// SECTION("addition_assignment_") {
//         SECTION("scalar") {
//             scalar_buffer scalar2(eigen_scalar, scalar_layout, alloc0);
//             scalar2.value()() = 42.0;

//             auto s        = scalar("");
//             auto pscalar2 = &(scalar2.addition_assignment("", s, s));

//             scalar_buffer scalar_corr(eigen_scalar, scalar_layout, alloc0);
//             scalar_corr.value()() = 20.0;
//             REQUIRE(pscalar2 == &scalar2);
//             REQUIRE(scalar2 == scalar_corr);
//         }

//         SECTION("vector") {
//             auto vector2 = testing::eigen_vector<TestType>();

//             auto vi       = vector("i");
//             auto pvector2 = &(vector2.addition_assignment("i", vi, vi));

//             vector_buffer vector_corr(eigen_vector, vector_layout, alloc1);
//             vector_corr.value()(0) = 20.0;
//             vector_corr.value()(1) = 40.0;

//             REQUIRE(pvector2 == &vector2);
//             REQUIRE(vector2 == vector_corr);
//         }

//         SECTION("matrix : no permutation") {
//             auto matrix2 = testing::eigen_matrix<TestType>();

//             auto mij      = matrix("i,j");
//             auto pmatrix2 = &(matrix2.addition_assignment("i,j", mij, mij));

//             matrix_buffer matrix_corr(eigen_matrix, matrix_layout, alloc2);

//             matrix_corr.value()(0, 0) = 20.0;
//             matrix_corr.value()(0, 1) = 40.0;
//             matrix_corr.value()(0, 2) = 60.0;
//             matrix_corr.value()(1, 0) = 80.0;
//             matrix_corr.value()(1, 1) = 100.0;
//             matrix_corr.value()(1, 2) = 120.0;

//             REQUIRE(pmatrix2 == &matrix2);
//             REQUIRE(matrix2 == matrix_corr);
//         }

//         SECTION("matrix: permutations") {
//             auto matrix2 = testing::eigen_matrix<TestType>();
//             auto l       = testing::matrix_physical(3, 2);
//             std::array<int, 2> p10{1, 0};
//             auto eigen_matrix_t = eigen_matrix.shuffle(p10);
//             matrix_buffer matrix1(eigen_matrix_t, l, alloc2);

//             auto mij = matrix("i,j");
//             auto mji = matrix1("j,i");

//             matrix_buffer matrix_corr(eigen_matrix, matrix_layout, alloc2);

//             matrix_corr.value()(0, 0) = 20.0;
//             matrix_corr.value()(0, 1) = 40.0;
//             matrix_corr.value()(0, 2) = 60.0;
//             matrix_corr.value()(1, 0) = 80.0;
//             matrix_corr.value()(1, 1) = 100.0;
//             matrix_corr.value()(1, 2) = 120.0;

//             SECTION("permute this") {
//                 matrix2.addition_assignment("j,i", mij, mij);

//                 matrix_buffer corr(eigen_matrix_t, l, alloc2);
//                 corr.value()(0, 0) = 20.0;
//                 corr.value()(0, 1) = 80.0;
//                 corr.value()(1, 0) = 40.0;
//                 corr.value()(1, 1) = 100.0;
//                 corr.value()(2, 0) = 60.0;
//                 corr.value()(2, 1) = 120.0;

//                 REQUIRE(matrix2 == corr);
//             }

//             SECTION("permute LHS") {
//                 matrix2.addition_assignment("i,j", mji, mij);
//                 REQUIRE(matrix2 == matrix_corr);
//             }

//             SECTION("permute RHS") {
//                 matrix2.addition_assignment("i,j", mij, mji);
//                 REQUIRE(matrix2 == matrix_corr);
//             }
//         }

//         SECTION("tensor (must permute all)") {
//             auto tensor2 = testing::eigen_tensor3<TestType>();

//             std::array<int, 3> p102{1, 0, 2};
//             auto l102 = testing::tensor3_physical(2, 1, 3);
//             tensor_buffer tensor102(eigen_tensor.shuffle(p102), l102,
//             alloc3);

//             auto tijk = tensor("i,j,k");
//             auto tjik = tensor102("j,i,k");

//             tensor2.addition_assignment("k,j,i", tijk, tjik);

//             std::array<int, 3> p210{2, 1, 0};
//             auto l210 = testing::tensor3_physical(3, 2, 1);
//             tensor_buffer corr(eigen_tensor.shuffle(p210), l210, alloc3);
//             corr.value()(0, 0, 0) = 20.0;
//             corr.value()(0, 1, 0) = 80.0;
//             corr.value()(1, 0, 0) = 40.0;
//             corr.value()(1, 1, 0) = 100.0;
//             corr.value()(2, 0, 0) = 60.0;
//             corr.value()(2, 1, 0) = 120.0;
//             REQUIRE(tensor2 == corr);
//         }
//     }