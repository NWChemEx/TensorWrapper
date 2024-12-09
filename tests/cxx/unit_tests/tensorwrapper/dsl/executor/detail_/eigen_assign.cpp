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

#include "../../../helpers.hpp"
#include "../../../inputs.hpp"
#include "../../../testing/eigen_buffers.hpp"
#include <tensorwrapper/dsl/executor/detail_/eigen_assign.hpp>

using namespace tensorwrapper::dsl::executor::detail_;
using namespace tensorwrapper::testing;

TEST_CASE("EigenAssign") {
    auto scalar = eigen_scalar<double>();
    auto vector = eigen_vector<double>();
    auto matrix = eigen_matrix<double>();

    SECTION("Scalar == Scalar") {
        auto scalar2      = eigen_scalar<double>();
        scalar2.value()() = 1.0;
        EigenAssign functor;
        auto pscalar = &(functor.run(scalar, scalar2));
        REQUIRE(pscalar == &scalar);
        REQUIRE(scalar.value()() == 1.0);
    }

    SECTION("Vector == Vector") {
        auto vector2 = eigen_vector<double>();
        auto size    = vector2.value().size();
        for(std::size_t i = 0; i < size; ++i) vector2.value()(i) = 0.0;

        EigenAssign functor;
        auto pvector = &(functor.run(vector, vector2));
        REQUIRE(pvector == &vector);
        for(std::size_t i = 0; i < size; ++i) REQUIRE(vector.value()(i) == 0.0);
    }

    SECTION("Can assign to empty") {
        decltype(matrix) matrix2;
        EigenAssign functor;
        auto pmatrix2 = &(functor.run(matrix2, matrix));
        REQUIRE(pmatrix2 == &matrix2);
        REQUIRE(matrix2 == matrix);
    }
}
