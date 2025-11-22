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

TEMPLATE_LIST_TEST_CASE("MDBuffer", "", types::floating_point_types) {
    using buffer::MDBuffer;
    using buffer_type = MDBuffer::buffer_type;
    using shape_type  = typename MDBuffer::shape_type;
    using label_type  = typename MDBuffer::label_type;

    TestType one(1.0), two(2.0), three(3.0), four(4.0);
    std::vector<TestType> data = {one, two, three, four};

    shape_type scalar_shape({});
    shape_type vector_shape({4});
    shape_type matrix_shape({2, 2});

    MDBuffer defaulted;
    MDBuffer scalar(std::vector{one}, scalar_shape);
    MDBuffer vector(data, vector_shape);
    MDBuffer matrix(data, matrix_shape);

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

            REQUIRE_THROWS_AS(MDBuffer(data, scalar_shape),
                              std::invalid_argument);
        }

        SECTION("FloatBuffer ctor") {
            buffer_type buf(data);

            MDBuffer vector_buf(buf, vector_shape);
            REQUIRE(vector_buf == vector);

            MDBuffer matrix_buf(buf, matrix_shape);
            REQUIRE(matrix_buf == matrix);

            REQUIRE_THROWS_AS(MDBuffer(buf, scalar_shape),
                              std::invalid_argument);
        }

        SECTION("Copy ctor") {
            MDBuffer defaulted_copy(defaulted);
            REQUIRE(defaulted_copy == defaulted);

            MDBuffer scalar_copy(scalar);
            REQUIRE(scalar_copy == scalar);

            MDBuffer vector_copy(vector);
            REQUIRE(vector_copy == vector);

            MDBuffer matrix_copy(matrix);
            REQUIRE(matrix_copy == matrix);
        }

        SECTION("Move ctor") {
            MDBuffer defaulted_temp(defaulted);
            MDBuffer defaulted_move(std::move(defaulted_temp));
            REQUIRE(defaulted_move == defaulted);

            MDBuffer scalar_temp(scalar);
            MDBuffer scalar_move(std::move(scalar_temp));
            REQUIRE(scalar_move == scalar);

            MDBuffer vector_temp(vector);
            MDBuffer vector_move(std::move(vector_temp));
            REQUIRE(vector_move == vector);

            MDBuffer matrix_temp(matrix);
            MDBuffer matrix_move(std::move(matrix_temp));
            REQUIRE(matrix_move == matrix);
        }

        SECTION("Copy assignment") {
            MDBuffer defaulted_copy;
            auto pdefaulted_copy = &(defaulted_copy = defaulted);
            REQUIRE(defaulted_copy == defaulted);
            REQUIRE(pdefaulted_copy == &defaulted_copy);

            MDBuffer scalar_copy;
            auto pscalar_copy = &(scalar_copy = scalar);
            REQUIRE(scalar_copy == scalar);
            REQUIRE(pscalar_copy == &scalar_copy);

            MDBuffer vector_copy;
            auto pvector_copy = &(vector_copy = vector);
            REQUIRE(vector_copy == vector);
            REQUIRE(pvector_copy == &vector_copy);

            MDBuffer matrix_copy;
            auto pmatrix_copy = &(matrix_copy = matrix);
            REQUIRE(matrix_copy == matrix);
            REQUIRE(pmatrix_copy == &matrix_copy);
        }

        SECTION("Move assignment") {
            MDBuffer defaulted_temp(defaulted);
            MDBuffer defaulted_move;
            auto pdefaulted_move =
              &(defaulted_move = std::move(defaulted_temp));
            REQUIRE(defaulted_move == defaulted);
            REQUIRE(pdefaulted_move == &defaulted_move);

            MDBuffer scalar_temp(scalar);
            MDBuffer scalar_move;
            auto pscalar_move = &(scalar_move = std::move(scalar_temp));
            REQUIRE(scalar_move == scalar);
            REQUIRE(pscalar_move == &scalar_move);

            MDBuffer vector_temp(vector);
            MDBuffer vector_move;
            auto pvector_move = &(vector_move = std::move(vector_temp));
            REQUIRE(vector_move == vector);
            REQUIRE(pvector_move == &vector_move);

            MDBuffer matrix_temp(matrix);
            MDBuffer matrix_move;
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

        MDBuffer scalar_copy(std::vector{one}, scalar_shape);
        REQUIRE(scalar == scalar_copy);

        MDBuffer vector_copy(data, vector_shape);
        REQUIRE(vector == vector_copy);

        MDBuffer matrix_copy(data, matrix_shape);
        REQUIRE(matrix == matrix_copy);

        // Different ranks
        REQUIRE_FALSE(scalar == vector);
        REQUIRE_FALSE(vector == matrix);
        REQUIRE_FALSE(scalar == matrix);

        // Different shapes
        shape_type matrix_shape2({4, 1});
        REQUIRE_FALSE(scalar == MDBuffer(data, matrix_shape2));

        // Different values
        std::vector<TestType> diff_data = {two, three, four, one};
        MDBuffer scalar_diff(std::vector{two}, scalar_shape);
        REQUIRE_FALSE(scalar == scalar_diff);
        REQUIRE_FALSE(vector == MDBuffer(diff_data, vector_shape));
        REQUIRE_FALSE(matrix == MDBuffer(diff_data, matrix_shape));
    }

    SECTION("addition_assignment_") {
        SECTION("scalar") {
            label_type labels("");
            MDBuffer result;
            result.addition_assignment(labels, scalar(labels), scalar(labels));
            REQUIRE(result.shape() == scalar_shape);
            REQUIRE(result.get_elem({}) == TestType(2.0));
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
