/*
 * Copyright 2022 NWChemEx-Project
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

#include "../test_tensor.hpp"
#include <tensorwrapper/tensor/expression/expression_class.hpp>
#include <tensorwrapper/tensor/tensor_wrapper.hpp>

using namespace tensorwrapper::tensor;

/* Testing Notes
 *
 * - The majority of Expression is implemented by classes deriving from
 *   ExpressionPIMPL. Those classes are unit tested elsewhere and assumed to
 *   work for the purposes of the unit tests here.
 * - Compared to ExpressionPIMPL the main scenarios we need to test are: an
 *   empty PIMPL and a non-empty PIMPL. How we set the PIMPL is somewhat
 *   irrelevant because the PIMPL instances are all tested.
 * - Arguably the easiest way to make an Expression instance with a non-empty
 *   PIMPL is to label a tensor
 * - for the operators we just check that they throw
 */

TEMPLATE_LIST_TEST_CASE("Expression", "", testing::field_types) {
    using field_type      = TestType;
    using tensor_type     = TensorWrapper<field_type>;
    using expression_type = expression::Expression<field_type>;

    constexpr bool is_tot = std::is_same_v<TestType, field::Tensor>;

    const auto ij = is_tot ? "i,j" : "i;j";
    auto tensors  = testing::get_tensors<field_type>();
    auto a        = tensors.at(is_tot ? "vector-of-vectors" : "vector");
    auto la       = a(ij);

    expression_type empty;
    auto non_empty = la.expression();

    SECTION("CTors") {
        SECTION("Default") { REQUIRE(empty == expression_type{}); }

        SECTION("Value") {
            REQUIRE(non_empty.labels(ij) == ij);
            REQUIRE(non_empty.tensor(ij, a.shape(), a.allocator()) == a);
        }

        SECTION("Copy") {
            expression_type empty_copy(empty);
            REQUIRE(empty_copy == empty);

            expression_type non_empty_copy(non_empty);
            REQUIRE(non_empty_copy == non_empty);
        }

        SECTION("Move") {
            expression_type empty_move(std::move(empty));
            REQUIRE(empty_move == expression_type{});

            expression_type non_empty_copy(non_empty);
            expression_type non_empty_move(std::move(non_empty));
            REQUIRE(non_empty_move == non_empty_copy);
        }

        SECTION("Copy Assignment") {
            expression_type empty_copy;
            auto pempty_copy = &(empty_copy = empty);
            REQUIRE(pempty_copy == &empty_copy);
            REQUIRE(empty_copy == empty);

            expression_type non_empty_copy;
            auto pnon_empty_copy = &(non_empty_copy = non_empty);
            REQUIRE(pnon_empty_copy == &non_empty_copy);
            REQUIRE(non_empty_copy == non_empty);
        }

        SECTION("Move Assignment") {
            expression_type empty_move;
            auto pempty_move = &(empty_move = std::move(empty));
            REQUIRE(pempty_move == &empty_move);
            REQUIRE(empty_move == expression_type{});

            expression_type non_empty_copy(non_empty);
            expression_type non_empty_move;
            auto pnon_empty_move = &(non_empty_move = std::move(non_empty));
            REQUIRE(pnon_empty_move == &non_empty_move);
            REQUIRE(non_empty_move == non_empty_copy);
        }
    }

    SECTION("operator+") {
        REQUIRE_THROWS_AS(empty + non_empty, std::runtime_error);
        REQUIRE_THROWS_AS(non_empty + empty, std::runtime_error);
    }

    SECTION("operator-") {
        REQUIRE_THROWS_AS(empty - non_empty, std::runtime_error);
        REQUIRE_THROWS_AS(non_empty - empty, std::runtime_error);
    }

    SECTION("operator*") {
        REQUIRE_THROWS_AS(empty * 3.14, std::runtime_error);
    }

    SECTION("operator*") {
        REQUIRE_THROWS_AS(empty * non_empty, std::runtime_error);
        REQUIRE_THROWS_AS(non_empty * empty, std::runtime_error);
    }

    SECTION("labels()") {
        REQUIRE_THROWS_AS(empty.labels(ij), std::runtime_error);
        REQUIRE(non_empty.labels(ij) == ij);
    }

    SECTION("tensor()") {
        const auto& shape = a.shape();
        const auto& alloc = a.allocator();
        REQUIRE_THROWS_AS(empty.tensor(ij, shape, alloc), std::runtime_error);

        REQUIRE(non_empty.tensor(ij, shape, alloc) == a);
    }

    SECTION("is_empty") {
        REQUIRE(empty.is_empty());
        REQUIRE_FALSE(non_empty.is_empty());
    }

    SECTION("swap") {
        expression_type non_empty_copy(non_empty);
        empty.swap(non_empty);

        REQUIRE(non_empty == expression_type{});
        REQUIRE(empty == non_empty_copy);
    }

    SECTION("operator==/operator!=") {
        REQUIRE(empty == expression_type{});
        REQUIRE_FALSE(empty != expression_type{});

        REQUIRE(non_empty == a(ij).expression());
        REQUIRE_FALSE(non_empty != a(ij).expression());

        REQUIRE_FALSE(non_empty == empty);
        REQUIRE(non_empty != empty);
    }
}
