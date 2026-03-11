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
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/buffer/replicated_view.hpp>
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace buffer;

TEST_CASE("ReplicatedView") {
    using view_type       = ReplicatedView<Replicated>;
    using const_view_type = ReplicatedView<const Replicated>;
    using TestType        = double;
    auto pvector          = testing::eigen_vector<TestType>(4);
    auto& vector          = *pvector;
    // vector has values [0, 1, 2, 3] at indices 0, 1, 2, 3

    auto pmatrix = testing::eigen_matrix<TestType>(2, 2);
    auto& matrix = *pmatrix;
    // matrix has values: (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4

    view_type vector_view(vector, {1}, {3});
    const_view_type const_vector_view(vector, {1}, {3});
    // vector_view has values [1, 2] at indices 0, 1

    view_type matrix_view(matrix, {0, 1}, {2, 2});
    const_view_type const_matrix_view(matrix, {0, 1}, {2, 2});
    // matrix_view has values [2, 4] at indices (0,0), (1,0)

    SECTION("Slice Constructor") {
        // Slice indices 1:3 of vector -> slice index 0 maps to 1, slice index 1
        // maps to 2
        REQUIRE(vector_view.get_elem({0}) == 1.0);
        REQUIRE(vector_view.get_elem({1}) == 2.0);

        auto corr_layout = vector.layout().slice({1}, {3});
        REQUIRE(vector_view.layout().are_equal(corr_layout));

        REQUIRE(const_vector_view.get_elem({0}) == 1.0);
        REQUIRE(const_vector_view.get_elem({1}) == 2.0);
        REQUIRE(const_vector_view.layout().are_equal(corr_layout));

        // Slice first_elem={0,1}, last_elem={2,2} -> 2 rows, 1 column
        // slice {0,0} -> underlying {0,1} -> value 2
        // slice {1,0} -> underlying {1,1} -> value 4

        REQUIRE(matrix_view.get_elem({0, 0}) == 2.0);
        REQUIRE(matrix_view.get_elem({1, 0}) == 4.0);

        auto matrix_slice_layout = matrix.layout().slice({0, 1}, {2, 2});
        REQUIRE(matrix_view.layout().are_equal(matrix_slice_layout));
        REQUIRE(const_matrix_view.get_elem({0, 0}) == 2.0);
        REQUIRE(const_matrix_view.get_elem({1, 0}) == 4.0);
        REQUIRE(const_matrix_view.layout().are_equal(matrix_slice_layout));
    }

    SECTION("clone") {
        vector_view.set_elem({0}, 1.0);
        auto cloned = view_type(vector_view);
        REQUIRE(cloned.get_elem({0}) == 1.0);
        REQUIRE(cloned.get_elem({1}) == 2.0);
    }

    SECTION("get_elem") {
        REQUIRE(const_vector_view.get_elem({0}) == 1.0);
        REQUIRE(const_vector_view.get_elem({1}) == 2.0);

        view_type defaulted;
        REQUIRE_THROWS_AS(defaulted.get_elem({0}), std::runtime_error);
    }

    SECTION("set_elem") {
        vector_view.set_elem({0}, 99.0);
        REQUIRE(vector.get_elem({1}) == 99.0);
        REQUIRE(vector_view.get_elem({0}) == 99.0);

        view_type defaulted;
        REQUIRE_THROWS_AS(defaulted.set_elem({0}, 1.0), std::runtime_error);
    }

    SECTION("slice()") {
        auto vector_slice = vector_view.slice({1}, {2});
        // vector_slice has values [2] at index 0
        REQUIRE(vector_slice.layout().shape().size() == 1);
        REQUIRE(vector_slice.get_elem({0}) == 2.0);

        auto const_vector_slice = const_vector_view.slice({1}, {2});
        REQUIRE(const_vector_slice.layout().shape().size() == 1);
        REQUIRE(const_vector_slice.get_elem({0}) == 2.0);

        auto matrix_slice = matrix_view.slice({0, 0}, {1, 1});
        // matrix_slice has values [2] at index (0,0)
        REQUIRE(matrix_slice.layout().shape().size() == 1);
        REQUIRE(matrix_slice.get_elem({0, 0}) == 2.0);

        auto const_matrix_slice = const_matrix_view.slice({0, 0}, {1, 1});
        REQUIRE(const_matrix_slice.layout().shape().size() == 1);
        REQUIRE(const_matrix_slice.get_elem({0, 0}) == 2.0);

        TestType nine_nine(99.0);
        vector_slice.set_elem({0}, nine_nine);
        REQUIRE(vector.get_elem({2}) == nine_nine);
        matrix_slice.set_elem({0, 0}, nine_nine);
        REQUIRE(matrix.get_elem({0, 1}) == nine_nine);
    }

    SECTION("slice() const") {
        auto vector_slice = std::as_const(vector_view).slice({1}, {2});
        REQUIRE(vector_slice.layout().shape().size() == 1);
        REQUIRE(vector_slice.get_elem({0}) == 2.0);

        auto const_vector_slice =
          std::as_const(const_vector_view).slice({1}, {2});
        REQUIRE(const_vector_slice.layout().shape().size() == 1);
        REQUIRE(const_vector_slice.get_elem({0}) == 2.0);

        auto matrix_slice = std::as_const(matrix_view).slice({0, 0}, {1, 1});
        REQUIRE(matrix_slice.layout().shape().size() == 1);
        REQUIRE(matrix_slice.get_elem({0, 0}) == 2.0);

        auto const_matrix_slice =
          std::as_const(const_matrix_view).slice({0, 0}, {1, 1});
        REQUIRE(const_matrix_slice.layout().shape().size() == 1);
        REQUIRE(const_matrix_slice.get_elem({0, 0}) == 2.0);
    }
}
