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
#include "../helpers.hpp"
#include "../inputs.hpp"
#include <tensorwrapper/tensor/detail_/tensor_factory.hpp>
#include <tensorwrapper/tensor/detail_/tensor_pimpl.hpp>
#include <tensorwrapper/tensor/tensor_class.hpp>
using namespace tensorwrapper;

TEST_CASE("Tensor") {
    using detail_::TensorFactory;

    Tensor defaulted;
    Tensor scalar(testing::smooth_scalar());
    Tensor vector(testing::smooth_vector());

    // We know TensorFactory works from unit testing it
    auto scalar_corr = TensorFactory::construct(testing::smooth_scalar());
    auto& scalar_layout_corr = scalar_corr->logical_layout();
    auto& scalar_buffer_corr = scalar_corr->buffer();

    auto vector_corr = TensorFactory::construct(testing::smooth_vector());
    auto& vector_layout_corr = vector_corr->logical_layout();
    auto& vector_buffer_corr = vector_corr->buffer();

    SECTION("Ctors") {
        SECTION("Value") {
            REQUIRE(scalar.logical_layout().are_equal(scalar_layout_corr));
            REQUIRE(scalar.buffer().are_equal(scalar_buffer_corr));

            REQUIRE(vector.logical_layout().are_equal(vector_layout_corr));
            REQUIRE(vector.buffer().are_equal(vector_buffer_corr));
        }

        SECTION("scalar_il_type") {
            Tensor t(42.0);
            Tensor corr(testing::smooth_scalar());
            REQUIRE(t == corr);
        }

        SECTION("vector_il_type") {
            using vector_il_type = typename Tensor::vector_il_type;
            vector_il_type il{0.0, 1.0, 2.0, 3.0, 4.0};
            Tensor t(il);
            Tensor corr(testing::smooth_vector());
            REQUIRE(t == corr);
        }

        SECTION("matrix_il_type") {
            using matrix_il_type = typename Tensor::matrix_il_type;
            matrix_il_type il{{1.0, 2.0}, {3.0, 4.0}};
            Tensor t(il);
            Tensor corr(testing::smooth_matrix());
            REQUIRE(t == corr);
        }

        SECTION("tensor3_il_type") {
            using tensor3_il_type = typename Tensor::tensor3_il_type;
            tensor3_il_type il{{{1.0, 2.0}, {3.0, 4.0}},
                               {{5.0, 6.0}, {7.0, 8.0}}};
            Tensor t(il);
            Tensor corr(testing::smooth_tensor3());
            REQUIRE(t == corr);
        }

        SECTION("tensor4_il_type") {
            using tensor4_il_type = typename Tensor::tensor4_il_type;
            tensor4_il_type il{
              {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}},
              {{{9.0, 10.0}, {11.0, 12.0}}, {{13.0, 14.0}, {15.0, 16.0}}}};
            Tensor t(il);
            Tensor corr(testing::smooth_tensor4());
            REQUIRE(t == corr);
        }

        testing::test_copy_move_ctor_and_assignment(scalar, vector);
    }

    SECTION("logical_layout () const") {
        auto& scalar_layout = std::as_const(scalar).logical_layout();
        REQUIRE(scalar_layout.are_equal(scalar_layout_corr));

        auto& vector_layout = std::as_const(vector).logical_layout();
        REQUIRE(vector_layout.are_equal(vector_layout_corr));

        const auto& const_defaulted = defaulted;
        REQUIRE_THROWS_AS(const_defaulted.logical_layout(), std::runtime_error);
    }

    SECTION("buffer() const") {
        auto& scalar_buffer = std::as_const(scalar).buffer();
        REQUIRE(scalar_buffer.are_equal(scalar_buffer_corr));

        auto& vector_buffer = std::as_const(vector).buffer();
        REQUIRE(vector_buffer.are_equal(vector_buffer_corr));

        const auto& const_defaulted = defaulted;
        REQUIRE_THROWS_AS(const_defaulted.buffer(), std::runtime_error);
    }

    SECTION("swap") {
        Tensor scalar_copy(scalar);
        Tensor vector_copy(vector);

        scalar.swap(vector);

        REQUIRE(scalar == vector_copy);
        REQUIRE(vector == scalar_copy);
    }

    SECTION("operator==") {
        Tensor other_scalar(testing::smooth_scalar());
        Tensor other_vector(testing::smooth_vector());
        REQUIRE(scalar == other_scalar);
        REQUIRE(vector == other_vector);

        SECTION("Different layout") {
            auto vector_input = testing::smooth_vector();
            shape::Smooth alt_shape{5, 1};
            symmetry::Group g;
            sparsity::Pattern sparsity;
            auto p = std::make_unique<layout::Logical>(alt_shape, g, sparsity);
            vector_input.m_pshape = nullptr;
            vector_input.m_plogical.swap(p);
            REQUIRE_FALSE(vector == Tensor(std::move(vector_input)));
        }

        SECTION("Different buffer") {
            Tensor vector_alt(testing::smooth_vector_alt());
            REQUIRE_FALSE(vector == vector_alt);
        }
    }

    SECTION("operator!=") {
        // Implemented in terms of operator==, just spot check
        Tensor other_scalar(testing::smooth_scalar());

        REQUIRE_FALSE(scalar != other_scalar);
        REQUIRE(scalar != vector);
    }
}
