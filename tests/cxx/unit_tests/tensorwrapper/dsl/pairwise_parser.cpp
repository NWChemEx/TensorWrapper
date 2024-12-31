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

#include "../testing/testing.hpp"
#include <tensorwrapper/dsl/pairwise_parser.hpp>

using namespace tensorwrapper;

TEST_CASE("PairwiseParser<Tensor>") {
    Tensor scalar(testing::smooth_scalar());
    Tensor vector(testing::smooth_vector());
    Tensor matrix(testing::smooth_matrix());

    dsl::PairwiseParser<Tensor, std::string> p;

    SECTION("add") {
        Tensor t;

        SECTION("scalar") {
            auto rv          = p.dispatch(t(""), scalar("") + scalar(""));
            auto buffer      = testing::eigen_scalar<double>();
            buffer.value()() = 84.0;
            Tensor corr(scalar.logical_layout(), std::move(buffer));
            REQUIRE(rv == corr);
        }

        SECTION("Vector") {
            auto vi     = vector("i");
            auto rv     = p.dispatch(t("i"), vi + vi);
            auto buffer = testing::eigen_vector<double>();
            for(std::size_t i = 0; i < 5; ++i) buffer.value()(i) = i + i;
            Tensor corr(vector.logical_layout(), std::move(buffer));
            REQUIRE(rv == corr);
        }

        SECTION("Matrix : no permutation") {
            auto mij                  = matrix("i,j");
            auto x                    = mij + mij;
            auto rv                   = p.dispatch(t("i,j"), x);
            auto matrix_corr          = testing::eigen_matrix<double>();
            matrix_corr.value()(0, 0) = 2.0;
            matrix_corr.value()(0, 1) = 4.0;
            matrix_corr.value()(0, 2) = 6.0;
            matrix_corr.value()(1, 0) = 8.0;
            matrix_corr.value()(1, 1) = 10.0;
            matrix_corr.value()(1, 2) = 12.0;
            Tensor corr(matrix.logical_layout(), std::move(matrix_corr));
            std::cout << rv.buffer() << std::endl;
            REQUIRE(rv == corr);
        }
    }
}