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

#include "../testing/testing.hpp"
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;

TEST_CASE("LayoutCommon") {
    using shape_type  = shape::Smooth;
    using layout_type = layout::Physical;
    using size_type   = layout_type::size_type;
    using size_vector = std::vector<size_type>;

    shape_type scalar_shape{};
    shape_type vector_shape{3};
    shape_type matrix_shape{3, 5};
    shape_type tensor_shape{9, 4, 12};

    layout_type empty;
    layout_type scalar(scalar_shape);
    layout_type vector(vector_shape);
    layout_type matrix(matrix_shape);
    layout_type tensor(tensor_shape);

    // Spot check, just calls slice(range)
    SECTION("slice(initializer list)") {
        REQUIRE(empty.slice({}, {}) == empty);

        layout_type vector_corr(vector_shape.slice({1}, {2}));
        REQUIRE(vector.slice({1}, {2}) == vector_corr);

        layout_type matrix_corr(matrix_shape.slice({1, 2}, {3, 5}));
        REQUIRE(matrix.slice({1, 2}, {3, 5}) == matrix_corr);

        layout_type tensor_corr(tensor_shape.slice({3, 1, 5}, {7, 4, 10}));
        REQUIRE(tensor.slice({3, 1, 5}, {7, 4, 10}) == tensor_corr);
    }

    // Spot check, just calls slice(range)
    SECTION("slice(initializer list)") {
        size_vector i;
        REQUIRE(empty.slice(i, i) == empty);

        size_vector i1{1}, i2{2};
        layout_type vector_corr(vector_shape.slice({1}, {2}));
        REQUIRE(vector.slice(i1, i2) == vector_corr);

        size_vector i12{1, 2}, i35{3, 5};
        layout_type matrix_corr(matrix_shape.slice({1, 2}, {3, 5}));
        REQUIRE(matrix.slice(i12, i35) == matrix_corr);

        size_vector i315{3, 1, 5}, i7410{7, 4, 10};
        layout_type tensor_corr(tensor_shape.slice({3, 1, 5}, {7, 4, 10}));
        REQUIRE(tensor.slice(i315, i7410) == tensor_corr);
    }

    // Spot check, just dispatches to shape/group/pattern slice method
    SECTION("slice(initializer list)") {
        size_vector i;
        REQUIRE(empty.slice(i.begin(), i.end(), i.begin(), i.end()) == empty);

        size_vector i1{1}, i2{2};
        layout_type vector_corr(vector_shape.slice({1}, {2}));
        REQUIRE(vector.slice(i1.begin(), i1.end(), i2.begin(), i2.end()) ==
                vector_corr);

        size_vector i12{1, 2}, i35{3, 5};
        layout_type matrix_corr(matrix_shape.slice({1, 2}, {3, 5}));
        REQUIRE(matrix.slice(i12.begin(), i12.end(), i35.begin(), i35.end()) ==
                matrix_corr);

        size_vector i315{3, 1, 5}, i7410{7, 4, 10};
        layout_type tensor_corr(tensor_shape.slice({3, 1, 5}, {7, 4, 10}));
        REQUIRE(tensor.slice(i315.begin(), i315.end(), i7410.begin(),
                             i7410.end()) == tensor_corr);
    }
}
