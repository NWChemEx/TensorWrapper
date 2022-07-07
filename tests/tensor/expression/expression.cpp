#include <catch2/catch.hpp>
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

TEST_CASE("Expression<field::Scalar>") {
    using field_type      = field::Scalar;
    using tensor_type     = TensorWrapper<field_type>;
    using expression_type = expression::Expression<field_type>;

    const auto ij = "i,j";
    tensor_type a{{1.0, 2.0}, {3.0, 4.0}};
    auto la = a(ij);

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
