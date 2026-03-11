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

#include "../../testing/testing.hpp"
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/buffer/detail_/slice_pimpl.hpp>

using namespace tensorwrapper;
using namespace buffer;
using namespace buffer::detail_;

TEST_CASE("SlicePIMPL") {
    using slice_type       = SlicePIMPL<Replicated>;
    using const_slice_type = SlicePIMPL<const Replicated>;
    auto pvector           = testing::eigen_vector<double>(4);
    auto& vector           = *pvector;
    // vector has values [0, 1, 2, 3] at indices 0, 1, 2, 3

    auto pmatrix = testing::eigen_matrix<double>(2, 2);
    auto& matrix = *pmatrix;
    // matrix has values: (0,0)=1, (0,1)=2, (1,0)=3, (1,1)=4

    slice_type vector_slice(&vector, {1}, {3});
    const_slice_type const_vector_slice(&vector, {1}, {3});

    slice_type matrix_slice(&matrix, {0, 1}, {2, 2});
    const_slice_type const_matrix_slice(&matrix, {0, 1}, {2, 2});

    SECTION("Constructor") {
        // Slice indices 1:3 of vector -> slice index 0 maps to 1, slice index 1
        // maps to 2
        REQUIRE(vector_slice.get_elem({0}) == 1.0);
        REQUIRE(vector_slice.get_elem({1}) == 2.0);

        REQUIRE(const_vector_slice.get_elem({0}) == 1.0);
        REQUIRE(const_vector_slice.get_elem({1}) == 2.0);

        // Slice first_elem={0,1}, last_elem={2,2} -> 2 rows, 1 column
        // slice {0,0} -> underlying {0,1} -> value 2
        // slice {1,0} -> underlying {1,1} -> value 4

        REQUIRE(matrix_slice.get_elem({0, 0}) == 2.0);
        REQUIRE(matrix_slice.get_elem({1, 0}) == 4.0);

        REQUIRE(const_matrix_slice.get_elem({0, 0}) == 2.0);
        REQUIRE(const_matrix_slice.get_elem({1, 0}) == 4.0);
    }

    SECTION("clone") {
        vector_slice.set_elem({0}, 1.0);
        auto cloned = vector_slice.clone();
        REQUIRE(cloned->get_elem({0}) == 1.0);
        REQUIRE(cloned->get_elem({1}) == 2.0);
    }

    SECTION("get_elem") {
        REQUIRE(const_vector_slice.get_elem({0}) == 1.0);
        REQUIRE(const_vector_slice.get_elem({1}) == 2.0);

        slice_type null_slice(nullptr, {0}, {1});
        REQUIRE_THROWS_AS(null_slice.get_elem({0}), std::runtime_error);
    }

    SECTION("set_elem") {
        vector_slice.set_elem({0}, 99.0);
        REQUIRE(vector.get_elem({1}) == 99.0);
        REQUIRE(vector_slice.get_elem({0}) == 99.0);

        slice_type null_slice(nullptr, {0}, {1});
        REQUIRE_THROWS_AS(null_slice.set_elem({0}, 1.0), std::runtime_error);
    }
}
