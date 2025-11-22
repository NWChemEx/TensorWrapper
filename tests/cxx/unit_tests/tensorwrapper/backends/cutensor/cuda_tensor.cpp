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
#include "../testing/contraction_assignment.hpp"
#include <tensorwrapper/backends/cutensor/cuda_tensor.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::backends::cutensor;

using supported_fp_types = std::tuple<float, double>;

TEMPLATE_LIST_TEST_CASE("CUDATensor", "", supported_fp_types) {
    using tensor_type = CUDATensor<TestType>;
    using shape_type  = typename tensor_type::shape_type;
    using label_type  = typename tensor_type::label_type;

    std::vector<TestType> data(16);
    for(std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<TestType>(i);

    std::span<TestType> data_span(data.data(), data.size());

    shape_type scalar_shape({});
    shape_type vector_shape({16});
    shape_type matrix_shape({4, 4});
    shape_type tensor3_shape({2, 2, 4});
    shape_type tensor4_shape({2, 2, 2, 2});

    tensor_type scalar(data_span, scalar_shape);
    tensor_type vector(data_span, vector_shape);
    tensor_type matrix(data_span, matrix_shape);
    tensor_type tensor3(data_span, tensor3_shape);
    tensor_type tensor4(data_span, tensor4_shape);

    SECTION("rank") {
        REQUIRE(scalar.rank() == 0);
        REQUIRE(vector.rank() == 1);
        REQUIRE(matrix.rank() == 2);
        REQUIRE(tensor3.rank() == 3);
        REQUIRE(tensor4.rank() == 4);
    }

    SECTION("size") {
        REQUIRE(scalar.size() == 1);
        REQUIRE(vector.size() == 16);
        REQUIRE(matrix.size() == 16);
        REQUIRE(tensor3.size() == 16);
        REQUIRE(tensor4.size() == 16);
    }

    SECTION("shape") {
        REQUIRE(scalar.shape() == scalar_shape);
        REQUIRE(vector.shape() == vector_shape);
        REQUIRE(matrix.shape() == matrix_shape);
        REQUIRE(tensor3.shape() == tensor3_shape);
        REQUIRE(tensor4.shape() == tensor4_shape);
    }

    SECTION("data()") {
        REQUIRE(scalar.data() == data.data());
        REQUIRE(vector.data() == data.data());
        REQUIRE(matrix.data() == data.data());
        REQUIRE(tensor3.data() == data.data());
        REQUIRE(tensor4.data() == data.data());
    }

    SECTION("data() const") {
        REQUIRE(std::as_const(scalar).data() == data.data());
        REQUIRE(std::as_const(vector).data() == data.data());
        REQUIRE(std::as_const(matrix).data() == data.data());
        REQUIRE(std::as_const(tensor3).data() == data.data());
        REQUIRE(std::as_const(tensor4).data() == data.data());
    }

    SECTION("contraction_assignment") {
#ifdef ENABLE_CUTESNSOR
        testing::contraction_assignment<tensor_type>();
#else
        label_type label("");
        REQUIRE_THROWS_AS(
          scalar.contraction_assignment(label, label, label, scalar, scalar),
          std::runtime_error);
#endif
    }
}
