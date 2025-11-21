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
        SECTION("Default ctor") { REQUIRE(defaulted.size() == 0); }

        SECTION("vector ctor") {}

        SECTION("FloatBuffer ctor") {
            buffer_type buf(data);
            REQUIRE_THROWS_AS(MDBuffer(buf, scalar_shape),
                              std::invalid_argument);
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

        REQUIRE(vector.get_elem({0}) == one);
        REQUIRE(vector.get_elem({1}) == two);
        REQUIRE(vector.get_elem({2}) == three);
        REQUIRE(vector.get_elem({3}) == four);

        REQUIRE(matrix.get_elem({0, 0}) == one);
        REQUIRE(matrix.get_elem({0, 1}) == two);
        REQUIRE(matrix.get_elem({1, 0}) == three);
        REQUIRE(matrix.get_elem({1, 1}) == four);
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
}
