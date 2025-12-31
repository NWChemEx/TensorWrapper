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

#include "../testing/testing.hpp"
#include <tensorwrapper/buffer/mdbuffer.hpp>
#include <tensorwrapper/types/floating_point.hpp>

using namespace tensorwrapper;

/* Testing notes:
 *
 * The various operations (addition_assignment, etc.) are not exhaustively
 * tested here. These operations are implemented via visitors that dispatch to
 * various backends. The visitors themselves are tested in their own unit tests.
 * Here we assume the visitors work and spot check a couple of operations for
 * to help catch any integration issues.
 */

TEMPLATE_LIST_TEST_CASE("Contiguous", "", types::floating_point_types) {
    using buffer::Contiguous;
    using buffer_type = Contiguous::buffer_type;
    using shape_type  = typename Contiguous::shape_type;
    using label_type  = typename Contiguous::label_type;

    TestType one(1.0), two(2.0), three(3.0), four(4.0);
    std::vector<TestType> data = {one, two, three, four};

    shape_type scalar_shape({});
    shape_type vector_shape({4});
    shape_type matrix_shape({2, 2});

    Contiguous defaulted;
    Contiguous scalar(std::vector{one}, scalar_shape);
    Contiguous vector(data, vector_shape);
    Contiguous matrix(data, matrix_shape);

    SECTION("Ctors and assignment") {
        SECTION("Default ctor") {
            REQUIRE(defaulted.size() == 0);
            REQUIRE(defaulted.shape() == shape_type());
        }

        SECTION("vector ctor") {
            REQUIRE(scalar.size() == 1);
            REQUIRE(scalar.shape() == scalar_shape);
            REQUIRE(scalar.get_elem({}) == one);

            REQUIRE(vector.size() == 4);
            REQUIRE(vector.shape() == vector_shape);
            REQUIRE(vector.get_elem({0}) == one);
            REQUIRE(vector.get_elem({1}) == two);
            REQUIRE(vector.get_elem({2}) == three);
            REQUIRE(vector.get_elem({3}) == four);

            REQUIRE(matrix.size() == 4);
            REQUIRE(matrix.shape() == matrix_shape);
            REQUIRE(matrix.get_elem({0, 0}) == one);
            REQUIRE(matrix.get_elem({0, 1}) == two);
            REQUIRE(matrix.get_elem({1, 0}) == three);
            REQUIRE(matrix.get_elem({1, 1}) == four);

            REQUIRE_THROWS_AS(Contiguous(data, scalar_shape),
                              std::invalid_argument);
        }

        SECTION("FloatBuffer ctor") {
            buffer_type buf(data);

            Contiguous vector_buf(buf, vector_shape);
            REQUIRE(vector_buf == vector);

            Contiguous matrix_buf(buf, matrix_shape);
            REQUIRE(matrix_buf == matrix);

            REQUIRE_THROWS_AS(Contiguous(buf, scalar_shape),
                              std::invalid_argument);
        }

        SECTION("Copy ctor") {
            Contiguous defaulted_copy(defaulted);
            REQUIRE(defaulted_copy == defaulted);

            Contiguous scalar_copy(scalar);
            REQUIRE(scalar_copy == scalar);

            Contiguous vector_copy(vector);
            REQUIRE(vector_copy == vector);

            Contiguous matrix_copy(matrix);
            REQUIRE(matrix_copy == matrix);
        }

        SECTION("Move ctor") {
            Contiguous defaulted_temp(defaulted);
            Contiguous defaulted_move(std::move(defaulted_temp));
            REQUIRE(defaulted_move == defaulted);

            Contiguous scalar_temp(scalar);
            Contiguous scalar_move(std::move(scalar_temp));
            REQUIRE(scalar_move == scalar);

            Contiguous vector_temp(vector);
            Contiguous vector_move(std::move(vector_temp));
            REQUIRE(vector_move == vector);

            Contiguous matrix_temp(matrix);
            Contiguous matrix_move(std::move(matrix_temp));
            REQUIRE(matrix_move == matrix);
        }

        SECTION("Copy assignment") {
            Contiguous defaulted_copy;
            auto pdefaulted_copy = &(defaulted_copy = defaulted);
            REQUIRE(defaulted_copy == defaulted);
            REQUIRE(pdefaulted_copy == &defaulted_copy);

            Contiguous scalar_copy;
            auto pscalar_copy = &(scalar_copy = scalar);
            REQUIRE(scalar_copy == scalar);
            REQUIRE(pscalar_copy == &scalar_copy);

            Contiguous vector_copy;
            auto pvector_copy = &(vector_copy = vector);
            REQUIRE(vector_copy == vector);
            REQUIRE(pvector_copy == &vector_copy);

            Contiguous matrix_copy;
            auto pmatrix_copy = &(matrix_copy = matrix);
            REQUIRE(matrix_copy == matrix);
            REQUIRE(pmatrix_copy == &matrix_copy);
        }

        SECTION("Move assignment") {
            Contiguous defaulted_temp(defaulted);
            Contiguous defaulted_move;
            auto pdefaulted_move =
              &(defaulted_move = std::move(defaulted_temp));
            REQUIRE(defaulted_move == defaulted);
            REQUIRE(pdefaulted_move == &defaulted_move);

            Contiguous scalar_temp(scalar);
            Contiguous scalar_move;
            auto pscalar_move = &(scalar_move = std::move(scalar_temp));
            REQUIRE(scalar_move == scalar);
            REQUIRE(pscalar_move == &scalar_move);

            Contiguous vector_temp(vector);
            Contiguous vector_move;
            auto pvector_move = &(vector_move = std::move(vector_temp));
            REQUIRE(vector_move == vector);
            REQUIRE(pvector_move == &vector_move);

            Contiguous matrix_temp(matrix);
            Contiguous matrix_move;
            auto pmatrix_move = &(matrix_move = std::move(matrix_temp));
            REQUIRE(matrix_move == matrix);
            REQUIRE(pmatrix_move == &matrix_move);
        }
    }

    SECTION("shape") {
        REQUIRE(defaulted.shape() == shape_type());
        REQUIRE(scalar.shape() == scalar_shape);
        REQUIRE(vector.shape() == vector_shape);
        REQUIRE(matrix.shape() == matrix_shape);
    }

    SECTION("size") {
        REQUIRE(defaulted.size() == 0);
        REQUIRE(scalar.size() == 1);
        REQUIRE(vector.size() == 4);
        REQUIRE(matrix.size() == 4);
    }

    SECTION("get_elem") {
        REQUIRE_THROWS_AS(defaulted.get_elem({}), std::out_of_range);

        REQUIRE(scalar.get_elem({}) == one);
        REQUIRE_THROWS_AS(scalar.get_elem({0}), std::out_of_range);

        REQUIRE(vector.get_elem({0}) == one);
        REQUIRE(vector.get_elem({1}) == two);
        REQUIRE(vector.get_elem({2}) == three);
        REQUIRE(vector.get_elem({3}) == four);
        REQUIRE_THROWS_AS(vector.get_elem({4}), std::out_of_range);

        REQUIRE(matrix.get_elem({0, 0}) == one);
        REQUIRE(matrix.get_elem({0, 1}) == two);
        REQUIRE(matrix.get_elem({1, 0}) == three);
        REQUIRE(matrix.get_elem({1, 1}) == four);
        REQUIRE_THROWS_AS(matrix.get_elem({2, 0}), std::out_of_range);
    }

    SECTION("set_elem") {
        REQUIRE_THROWS_AS(defaulted.set_elem({}, one), std::out_of_range);

        REQUIRE(scalar.get_elem({}) != two);
        scalar.set_elem({}, two);
        REQUIRE(scalar.get_elem({}) == two);

        REQUIRE(vector.get_elem({2}) != four);
        vector.set_elem({2}, four);
        REQUIRE(vector.get_elem({2}) == four);

        REQUIRE(matrix.get_elem({1, 0}) != one);
        matrix.set_elem({1, 0}, one);
        REQUIRE(matrix.get_elem({1, 0}) == one);
    }

    SECTION("operator==") {
        // Same object
        REQUIRE(defaulted == defaulted);

        Contiguous scalar_copy(std::vector{one}, scalar_shape);
        REQUIRE(scalar == scalar_copy);

        Contiguous vector_copy(data, vector_shape);
        REQUIRE(vector == vector_copy);

        Contiguous matrix_copy(data, matrix_shape);
        REQUIRE(matrix == matrix_copy);

        // Different ranks
        REQUIRE_FALSE(scalar == vector);
        REQUIRE_FALSE(vector == matrix);
        REQUIRE_FALSE(scalar == matrix);

        // Different shapes
        shape_type matrix_shape2({4, 1});
        REQUIRE_FALSE(scalar == Contiguous(data, matrix_shape2));

        // Different values
        std::vector<TestType> diff_data = {two, three, four, one};
        Contiguous scalar_diff(std::vector{two}, scalar_shape);
        REQUIRE_FALSE(scalar == scalar_diff);
        REQUIRE_FALSE(vector == Contiguous(diff_data, vector_shape));
        REQUIRE_FALSE(matrix == Contiguous(diff_data, matrix_shape));
    }

    SECTION("addition_assignment_") {
        SECTION("scalar") {
            label_type labels("");
            Contiguous result;
            result.addition_assignment(labels, scalar(labels), scalar(labels));
            REQUIRE(result.shape() == scalar_shape);
            REQUIRE(result.get_elem({}) == TestType(2.0));
        }

        SECTION("vector") {
            label_type labels("i");
            Contiguous result;
            result.addition_assignment(labels, vector(labels), vector(labels));
            REQUIRE(result.shape() == vector_shape);
            REQUIRE(result.get_elem({0}) == TestType(2.0));
            REQUIRE(result.get_elem({1}) == TestType(4.0));
            REQUIRE(result.get_elem({2}) == TestType(6.0));
            REQUIRE(result.get_elem({3}) == TestType(8.0));
        }

        SECTION("matrix") {
            label_type labels("i,j");
            Contiguous result;
            result.addition_assignment(labels, matrix(labels), matrix(labels));
            REQUIRE(result.shape() == matrix_shape);
            REQUIRE(result.get_elem({0, 0}) == TestType(2.0));
            REQUIRE(result.get_elem({0, 1}) == TestType(4.0));
            REQUIRE(result.get_elem({1, 0}) == TestType(6.0));
            REQUIRE(result.get_elem({1, 1}) == TestType(8.0));
        }
    }

    SECTION("subtraction_assignment_") {
        SECTION("scalar") {
            label_type labels("");
            Contiguous result;
            result.subtraction_assignment(labels, scalar(labels),
                                          scalar(labels));
            REQUIRE(result.shape() == scalar_shape);
            REQUIRE(result.get_elem({}) == TestType(0.0));
        }

        SECTION("vector") {
            label_type labels("i");
            Contiguous result;
            result.subtraction_assignment(labels, vector(labels),
                                          vector(labels));
            REQUIRE(result.shape() == vector_shape);
            REQUIRE(result.get_elem({0}) == TestType(0.0));
            REQUIRE(result.get_elem({1}) == TestType(0.0));
            REQUIRE(result.get_elem({2}) == TestType(0.0));
            REQUIRE(result.get_elem({3}) == TestType(0.0));
        }

        SECTION("matrix") {
            label_type labels("i,j");
            Contiguous result;
            result.subtraction_assignment(labels, matrix(labels),
                                          matrix(labels));
            REQUIRE(result.shape() == matrix_shape);
            REQUIRE(result.get_elem({0, 0}) == TestType(0.0));
            REQUIRE(result.get_elem({0, 1}) == TestType(0.0));
            REQUIRE(result.get_elem({1, 0}) == TestType(0.0));
            REQUIRE(result.get_elem({1, 1}) == TestType(0.0));
        }
    }

    SECTION("multiplication_assignment_") {
        // N.b., dispatching among hadamard, contraction, etc. is the visitor's
        // responsibility and happens there. Here we just test hadamard.

        SECTION("scalar") {
            label_type labels("");
            Contiguous result;
            result.multiplication_assignment(labels, scalar(labels),
                                             scalar(labels));
            REQUIRE(result.shape() == scalar_shape);
            REQUIRE(result.get_elem({}) == TestType(1.0));
        }

        SECTION("vector") {
            label_type labels("i");
            Contiguous result;
            result.multiplication_assignment(labels, vector(labels),
                                             vector(labels));
            REQUIRE(result.shape() == vector_shape);
            REQUIRE(result.get_elem({0}) == TestType(1.0));
            REQUIRE(result.get_elem({1}) == TestType(4.0));
            REQUIRE(result.get_elem({2}) == TestType(9.0));
            REQUIRE(result.get_elem({3}) == TestType(16.0));
        }

        SECTION("matrix") {
            label_type labels("i,j");
            Contiguous result;
            result.multiplication_assignment(labels, matrix(labels),
                                             matrix(labels));
            REQUIRE(result.shape() == matrix_shape);
            REQUIRE(result.get_elem({0, 0}) == TestType(1.0));
            REQUIRE(result.get_elem({0, 1}) == TestType(4.0));
            REQUIRE(result.get_elem({1, 0}) == TestType(9.0));
            REQUIRE(result.get_elem({1, 1}) == TestType(16.0));
        }
    }

    SECTION("scalar_multiplication_") {
        // TODO: Test with other scalar types when public API supports it
        using scalar_type = double;
        scalar_type scalar_value_{2.0};
        TestType scalar_value(scalar_value_);
        SECTION("scalar") {
            label_type labels("");
            Contiguous result;
            result.scalar_multiplication(labels, scalar_value_, scalar(labels));
            REQUIRE(result.shape() == scalar_shape);
            REQUIRE(result.get_elem({}) == TestType(1.0) * scalar_value);
        }

        SECTION("vector") {
            label_type labels("i");
            Contiguous result;
            result.scalar_multiplication(labels, scalar_value_, vector(labels));
            REQUIRE(result.shape() == vector_shape);
            REQUIRE(result.get_elem({0}) == TestType(1.0) * scalar_value);
            REQUIRE(result.get_elem({1}) == TestType(2.0) * scalar_value);
            REQUIRE(result.get_elem({2}) == TestType(3.0) * scalar_value);
            REQUIRE(result.get_elem({3}) == TestType(4.0) * scalar_value);
        }

        SECTION("matrix") {
            label_type rhs_labels("i,j");
            label_type lhs_labels("j,i");
            Contiguous result;
            result.scalar_multiplication(lhs_labels, scalar_value_,
                                         matrix(rhs_labels));
            REQUIRE(result.shape() == matrix_shape);
            REQUIRE(result.get_elem({0, 0}) == TestType(1.0) * scalar_value);
            REQUIRE(result.get_elem({0, 1}) == TestType(3.0) * scalar_value);
            REQUIRE(result.get_elem({1, 0}) == TestType(2.0) * scalar_value);
            REQUIRE(result.get_elem({1, 1}) == TestType(4.0) * scalar_value);
        }
    }

    SECTION("permute_assignment_") {
        SECTION("scalar") {
            label_type labels("");
            Contiguous result;
            result.permute_assignment(labels, scalar(labels));
            REQUIRE(result.shape() == scalar_shape);
            REQUIRE(result.get_elem({}) == TestType(1.0));
        }

        SECTION("vector") {
            label_type labels("i");
            Contiguous result;
            result.permute_assignment(labels, vector(labels));
            REQUIRE(result.shape() == vector_shape);
            REQUIRE(result.get_elem({0}) == TestType(1.0));
            REQUIRE(result.get_elem({1}) == TestType(2.0));
            REQUIRE(result.get_elem({2}) == TestType(3.0));
            REQUIRE(result.get_elem({3}) == TestType(4.0));
        }

        SECTION("matrix") {
            label_type rhs_labels("i,j");
            label_type lhs_labels("j,i");
            Contiguous result;
            result.permute_assignment(lhs_labels, matrix(rhs_labels));
            REQUIRE(result.shape() == matrix_shape);
            REQUIRE(result.get_elem({0, 0}) == TestType(1.0));
            REQUIRE(result.get_elem({0, 1}) == TestType(3.0));
            REQUIRE(result.get_elem({1, 0}) == TestType(2.0));
            REQUIRE(result.get_elem({1, 1}) == TestType(4.0));
        }
    }

    SECTION("to_string") {
        REQUIRE(defaulted.to_string().empty());
        REQUIRE_FALSE(scalar.to_string().empty());
        REQUIRE_FALSE(vector.to_string().empty());
        REQUIRE_FALSE(matrix.to_string().empty());
    }

    SECTION("add_to_stream") {
        std::stringstream ss;
        SECTION("defaulted") {
            defaulted.add_to_stream(ss);
            REQUIRE(ss.str().empty());
        }
        SECTION("scalar") {
            scalar.add_to_stream(ss);
            REQUIRE_FALSE(ss.str().empty());
        }
        SECTION("vector") {
            vector.add_to_stream(ss);
            REQUIRE_FALSE(ss.str().empty());
        }
        SECTION("matrix") {
            matrix.add_to_stream(ss);
            REQUIRE_FALSE(ss.str().empty());
        }
    }
}
