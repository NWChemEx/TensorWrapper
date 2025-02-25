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

// SECTION("scalar_multiplication_") {
//         SECTION("scalar") {
//             auto scalar2      = testing::eigen_scalar<TestType>();
//             scalar2.value()() = 42.0;

//             auto s        = scalar("");
//             auto pscalar2 = &(scalar2.scalar_multiplication("", 2.0, s));

//             auto corr      = testing::eigen_scalar<TestType>();
//             corr.value()() = 20.0;
//             REQUIRE(pscalar2 == &scalar2);
//             REQUIRE(scalar2 == corr);
//         }

//         SECTION("vector") {
//             auto vector2 = testing::eigen_vector<TestType>();

//             auto vi       = vector("i");
//             auto pvector2 = &(vector2.scalar_multiplication("i", 2.0, vi));

//             auto corr       = testing::eigen_vector<TestType>(2);
//             corr.value()(0) = 20.0;
//             corr.value()(1) = 40.0;

//             REQUIRE(pvector2 == &vector2);
//             REQUIRE(vector2 == corr);
//         }

//         SECTION("matrix : no permutation") {
//             auto matrix2 = testing::eigen_matrix<TestType>();

//             auto mij = matrix("i,j");
//             auto p   = &(matrix2.scalar_multiplication("i,j", 2.0, mij));

//             auto corr          = testing::eigen_matrix<TestType>(2, 3);
//             corr.value()(0, 0) = 20.0;
//             corr.value()(0, 1) = 40.0;
//             corr.value()(0, 2) = 60.0;
//             corr.value()(1, 0) = 80.0;
//             corr.value()(1, 1) = 100.0;
//             corr.value()(1, 2) = 120.0;

//             REQUIRE(p == &matrix2);
//             REQUIRE(matrix2 == corr);
//         }

//         SECTION("matrix: permutation") {
//             auto matrix2 = testing::eigen_matrix<TestType>();
//             auto mij     = matrix("i,j");
//             auto p       = &(matrix2.scalar_multiplication("j,i", 2.0, mij));

//             auto corr          = testing::eigen_matrix<TestType>(3, 2);
//             corr.value()(0, 0) = 20.0;
//             corr.value()(1, 0) = 40.0;
//             corr.value()(2, 0) = 60.0;
//             corr.value()(0, 1) = 80.0;
//             corr.value()(1, 1) = 100.0;
//             corr.value()(2, 1) = 120.0;
//             REQUIRE(p == &matrix2);
//             compare_eigen<TestType>(corr.value(), matrix2.value());
//         }
//     }