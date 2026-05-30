/*
 * Copyright 2026 NWChemEx-Project
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

#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/generate/random_orthogonal_matrix.hpp>
#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <tensorwrapper/utilities/diagonal_matrix.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::buffer;
using namespace tensorwrapper::generate;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;

TEMPLATE_LIST_TEST_CASE("random_orthogonal_matrix", "",
                        types::floating_point_types) {
    SECTION("shape") {
        auto gen = make_rng(7);
        auto Q   = random_orthogonal_matrix<TestType>(3, gen);
        auto buf = make_contiguous(Q.buffer());
        REQUIRE(buf.shape().rank() == 2);
        REQUIRE(buf.shape().extent(0) == 3);
        REQUIRE(buf.shape().extent(1) == 3);
    }

    SECTION("orthogonality") {
        auto gen = make_rng(11);
        auto Q   = random_orthogonal_matrix<TestType>(3, gen);
        Tensor product;
        product("i,k") = Q("i,j") * Q("k,j");

        auto ones = make_tensor(
          {3}, std::vector<TestType>{TestType{1}, TestType{1}, TestType{1}});
        auto ident           = diagonal_matrix(ones);
        constexpr double tol = std::is_same_v<TestType, float> ||
                                   std::is_same_v<TestType, types::ufloat> ||
                                   std::is_same_v<TestType, types::ifloat> ?
                                 1e-5 :
                                 1e-12;
        REQUIRE(approximately_equal(product, ident, tol));
    }

    SECTION("deterministic for fixed seed") {
        auto gen1 = make_rng(99);
        auto gen2 = make_rng(99);
        auto Q1   = random_orthogonal_matrix<TestType>(4, gen1);
        auto Q2   = random_orthogonal_matrix<TestType>(4, gen2);
        REQUIRE(approximately_equal(Q1, Q2));
    }
}
