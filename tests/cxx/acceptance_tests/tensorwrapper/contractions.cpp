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

#include "acceptance_testing.hpp"

using namespace tensorwrapper;
using namespace operations;
TEST_CASE("Contractions") {
    Tensor defaulted;

    Tensor scalar_0(1.23);
    Tensor scalar_1(2.34);

    Tensor vector_0{1.23, 2.34, 3.45};
    Tensor vector_1{4.56, 5.67, 6.78};

    Tensor matrix_0{{1.23, 2.34}, {3.45, 4.56}};
    Tensor matrix_1{{5.67, 6.78}, {7.89, 8.90}};

    Tensor tensor3_0{{{1.1, 2.2}, {3.3, 4.4}}, {{5.5, 6.6}, {7.7, 8.8}}};
    Tensor tensor3_1{{{9.9, 10.10}, {11.11, 12.12}},
                     {{13.13, 14.14}, {15.15, 16.16}}};

    Tensor tensor4_0{
      {{{1.1, 2.2}, {3.3, 4.4}}, {{5.5, 6.6}, {7.7, 8.8}}},
      {{{9.9, 10.10}, {11.11, 12.12}}, {{13.13, 14.14}, {15.15, 16.16}}}};
    Tensor tensor4_1{
      {{{17.17, 18.18}, {19.19, 20.20}}, {{21.21, 22.22}, {23.23, 24.24}}},
      {{{25.25, 26.26}, {27.27, 28.28}}, {{29.29, 30.30}, {31.31, 32.32}}}};

    SECTION("LHS == matrix") {
        SECTION("RHS == tensor3") {
            SECTION("ij,jkl->ikl") {
                defaulted("i,k,l") = matrix_0("i,j") * tensor3_0("j,k,l");
                Tensor corr{
                  {{14.222999999999999, 18.15}, {22.076999999999998, 26.004}},
                  {{28.875, 37.686}, {46.497, 55.308}}};
                REQUIRE(approximately_equal(defaulted, corr, 1e-10));
            }
            SECTION("ij,ijk->k") {
                defaulted("k") = matrix_0("i,j") * tensor3_0("i,j,k");
                Tensor corr{63.162, 75.89999999999999};
                REQUIRE(approximately_equal(defaulted, corr, 1e-10));
            }
        }
    }
}
