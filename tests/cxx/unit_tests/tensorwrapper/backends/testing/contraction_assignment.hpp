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

#pragma once
#include <span>
#include <vector>

namespace tensorwrapper::testing {

template<typename ScalarType, typename VectorType = ScalarType,
         typename MatrixType = VectorType, typename Tensor3Type = MatrixType,
         typename Tensor4Type = Tensor3Type>
void contraction_assignment_tests() {
    using scalar_value_type  = typename ScalarType::value_type;
    using vector_value_type  = typename VectorType::value_type;
    using matrix_value_type  = typename MatrixType::value_type;
    using tensor3_value_type = typename Tensor3Type::value_type;
    using tensor4_value_type = typename Tensor4Type::value_type;

    using shape_type = typename ScalarType::shape_type;
    using label_type = typename ScalarType::label_type;

    std::vector<scalar_value_type> scalar_data(1, scalar_value_type(42.0));
    std::vector<vector_value_type> vector_data(2, vector_value_type(0.0));
    std::vector<matrix_value_type> matrix_data(4, matrix_value_type(0.0));
    std::vector<tensor3_value_type> tensor3_data(8, tensor3_value_type(0.0));
    std::vector<tensor4_value_type> tensor4_data(16, tensor4_value_type(0.0));

    for(std::size_t i = 0; i < vector_data.size(); ++i)
        vector_data[i] = scalar_value_type(i + 1.0);

    for(std::size_t i = 0; i < matrix_data.size(); ++i)
        matrix_data[i] = scalar_value_type(i + 1.0);

    for(std::size_t i = 0; i < tensor3_data.size(); ++i)
        tensor3_data[i] = scalar_value_type(i + 1.0);

    std::span<vector_value_type> vector_data_span(vector_data.data(),
                                                  vector_data.size());
    std::span<matrix_value_type> matrix_data_span(matrix_data.data(),
                                                  matrix_data.size());
    std::span<tensor3_value_type> tensor3_data_span(tensor3_data.data(),
                                                    tensor3_data.size());
    std::span<tensor4_value_type> tensor4_data_span(tensor4_data.data(),
                                                    tensor4_data.size());

    shape_type scalar_shape{};
    shape_type vector_shape{2};
    shape_type matrix_shape{2, 2};
    shape_type tensor3_shape{2, 2, 2};
    shape_type tensor4_shape{2, 2, 2, 2};

    ScalarType scalar(scalar_data, scalar_shape);
    VectorType vector(vector_data_span, vector_shape);
    MatrixType matrix(matrix_data_span, matrix_shape);
    Tensor3Type tensor3(tensor3_data_span, tensor3_shape);
    Tensor4Type tensor4(tensor4_data, shape_type{2, 2, 2, 2});

    SECTION("scalar,scalar->") {
        label_type o("");
        label_type l("");
        label_type r("");
        scalar.contraction_assignment(o, l, r, scalar, scalar);

        REQUIRE(scalar.get_elem({}) == scalar_value_type(42.0 * 42.0));
    }

    SECTION("i,i->") {
        label_type o("");
        label_type l("i");
        label_type r("i");
        scalar.contraction_assignment(o, l, r, vector, vector);
        REQUIRE(scalar.get_elem({}) == vector_value_type(5.0));
    }

    SECTION("i,ij->j") {
        label_type o("j");
        label_type l("i");
        label_type r("i,j");
        vector.contraction_assignment(o, l, r, vector, matrix);
        REQUIRE(vector.get_elem({0}) == vector_value_type(7.0));
        REQUIRE(vector.get_elem({1}) == vector_value_type(10.0));
    }

    SECTION("ij,ji->") {
        label_type o("");
        label_type l("i,j");
        label_type r("j,i");
        scalar.contraction_assignment(o, l, r, matrix, matrix);

        REQUIRE(scalar.get_elem({}) == matrix_value_type(29.0));
    }

    SECTION("ij,jk->ik") {
        label_type o("i,k");
        label_type l("i,j");
        label_type r("j,k");
        matrix.contraction_assignment(o, l, r, matrix, matrix);

        REQUIRE(matrix.get_elem({0, 0}) == matrix_value_type(7.0));
        REQUIRE(matrix.get_elem({0, 1}) == matrix_value_type(10.0));
        REQUIRE(matrix.get_elem({1, 0}) == matrix_value_type(15.0));
        REQUIRE(matrix.get_elem({1, 1}) == matrix_value_type(22.0));
    }

    SECTION("ijk,ijk->") {
        label_type o("");
        label_type l("i,j,k");
        label_type r("i,j,k");
        scalar.contraction_assignment(o, l, r, tensor3, tensor3);

        REQUIRE(scalar.get_elem({}) == scalar_value_type(204.0));
    }

    SECTION("ijk,jik->") {
        label_type o("");
        label_type l("i,j,k");
        label_type r("j,i,k");
        scalar.contraction_assignment(o, l, r, tensor3, tensor3);

        REQUIRE(scalar.get_elem({}) == scalar_value_type(196.0));
    }

    SECTION("ijk,jkl->il") {
        label_type o("i,l");
        label_type l("i,j,k");
        label_type r("j,k,l");
        matrix.contraction_assignment(o, l, r, tensor3, tensor3);

        REQUIRE(matrix.get_elem({0, 0}) == matrix_value_type(50.0));
        REQUIRE(matrix.get_elem({0, 1}) == matrix_value_type(60.0));
        REQUIRE(matrix.get_elem({1, 0}) == matrix_value_type(114.0));
        REQUIRE(matrix.get_elem({1, 1}) == matrix_value_type(140.0));
    }

    SECTION("ijk,jlk->il") {
        label_type o("i,l");
        label_type l("i,j,k");
        label_type r("j,l,k");
        matrix.contraction_assignment(o, l, r, tensor3, tensor3);

        REQUIRE(matrix.get_elem({0, 0}) == matrix_value_type(44.0));
        REQUIRE(matrix.get_elem({0, 1}) == matrix_value_type(64.0));
        REQUIRE(matrix.get_elem({1, 0}) == matrix_value_type(100.0));
        REQUIRE(matrix.get_elem({1, 1}) == matrix_value_type(152.0));
    }

    SECTION("ijk,jlk->li") {
        label_type o("l,i");
        label_type l("i,j,k");
        label_type r("j,l,k");
        matrix.contraction_assignment(o, l, r, tensor3, tensor3);

        REQUIRE(matrix.get_elem({0, 0}) == matrix_value_type(44.0));
        REQUIRE(matrix.get_elem({0, 1}) == matrix_value_type(100.0));
        REQUIRE(matrix.get_elem({1, 0}) == matrix_value_type(64.0));
        REQUIRE(matrix.get_elem({1, 1}) == matrix_value_type(152.0));
    }

    // SECTION("ijk,ljm->iklm") {

    SECTION("ijk,ljm->iklm") {
        label_type o("i,k,l,m");
        label_type l("i,j,k");
        label_type r("l,j,m");
        tensor4.contraction_assignment(o, l, r, tensor3, tensor3);

        REQUIRE(tensor4.get_elem({0, 0, 0, 0}) == tensor4_value_type(10.0));
        REQUIRE(tensor4.get_elem({0, 0, 0, 1}) == tensor4_value_type(14.0));
        REQUIRE(tensor4.get_elem({0, 0, 1, 0}) == tensor4_value_type(26.0));
        REQUIRE(tensor4.get_elem({0, 0, 1, 1}) == tensor4_value_type(30.0));
        REQUIRE(tensor4.get_elem({0, 1, 0, 0}) == tensor4_value_type(14.0));
        REQUIRE(tensor4.get_elem({0, 1, 0, 1}) == tensor4_value_type(20.0));
        REQUIRE(tensor4.get_elem({0, 1, 1, 0}) == tensor4_value_type(38.0));
        REQUIRE(tensor4.get_elem({0, 1, 1, 1}) == tensor4_value_type(44.0));
        REQUIRE(tensor4.get_elem({1, 0, 0, 0}) == tensor4_value_type(26.0));
        REQUIRE(tensor4.get_elem({1, 0, 0, 1}) == tensor4_value_type(38.0));
        REQUIRE(tensor4.get_elem({1, 0, 1, 0}) == tensor4_value_type(74.0));
        REQUIRE(tensor4.get_elem({1, 0, 1, 1}) == tensor4_value_type(86.0));
        REQUIRE(tensor4.get_elem({1, 1, 0, 0}) == tensor4_value_type(30.0));
        REQUIRE(tensor4.get_elem({1, 1, 0, 1}) == tensor4_value_type(44.0));
        REQUIRE(tensor4.get_elem({1, 1, 1, 0}) == tensor4_value_type(86.0));
        REQUIRE(tensor4.get_elem({1, 1, 1, 1}) == tensor4_value_type(100.0));
    }

    // SECTION("ij,jkl->ikl") {

    SECTION("ij,jkl->ikl") {
        label_type o("i,k,l");
        label_type l("i,j");
        label_type r("j,k,l");
        tensor3.contraction_assignment(o, l, r, matrix, tensor3);

        REQUIRE(tensor3.get_elem({0, 0, 0}) == tensor3_value_type(11.0));
        REQUIRE(tensor3.get_elem({0, 0, 1}) == tensor3_value_type(14.0));
        REQUIRE(tensor3.get_elem({0, 1, 0}) == tensor3_value_type(17.0));
        REQUIRE(tensor3.get_elem({0, 1, 1}) == tensor3_value_type(20.0));
        REQUIRE(tensor3.get_elem({1, 0, 0}) == tensor3_value_type(23.0));
        REQUIRE(tensor3.get_elem({1, 0, 1}) == tensor3_value_type(30.0));
        REQUIRE(tensor3.get_elem({1, 1, 0}) == tensor3_value_type(37.0));
        REQUIRE(tensor3.get_elem({1, 1, 1}) == tensor3_value_type(44.0));
    }
}
} // namespace tensorwrapper::testing
