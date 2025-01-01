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
    // Tensor scalar(testing::smooth_scalar());
    // Tensor vector(testing::smooth_vector());

    // dsl::PairwiseParser<Tensor, std::string> p;

    // SECTION("add") {
    //     Tensor t;

    //     SECTION("scalar") {
    //         auto rv = p.dispatch(t(""), scalar("") + scalar(""));
    //         REQUIRE(&rv.lhs() == &t);
    //         REQUIRE(rv.rhs() == "");

    //         auto buffer      = testing::eigen_scalar<double>();
    //         buffer.value()() = 84.0;
    //         Tensor corr(scalar.logical_layout(), std::move(buffer));
    //         REQUIRE(t == corr);
    //     }

    //     SECTION("Vector") {
    //         auto rv = p.dispatch(t("i"), vector("i") + vector("i"));
    //         REQUIRE(&rv.lhs() == &t);
    //         REQUIRE(rv.rhs() == "i");

    //         auto buffer = testing::eigen_vector<double>();
    //         for(std::size_t i = 0; i < 5; ++i) buffer.value()(i) = i + i;
    //         Tensor corr(t.logical_layout(), std::move(buffer));
    //         REQUIRE(t == corr);
    //     }
    // }
}