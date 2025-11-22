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
#include "../testing/addition_assignment.hpp"
#include "../testing/contraction_assignment.hpp"
#include "../testing/hadamard_assignment.hpp"
#include "../testing/permute_assignment.hpp"
#include "../testing/scalar_multiplication.hpp"
#include "../testing/subtraction_assignment.hpp"
#include <tensorwrapper/backends/eigen/eigen_tensor_impl.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::backends::eigen;

TEMPLATE_LIST_TEST_CASE("EigenTensorImpl", "", types::floating_point_types) {
    using scalar_type  = EigenTensorImpl<TestType, 0>;
    using vector_type  = EigenTensorImpl<TestType, 1>;
    using matrix_type  = EigenTensorImpl<TestType, 2>;
    using tensor3_type = EigenTensorImpl<TestType, 3>;
    using tensor4_type = EigenTensorImpl<TestType, 4>;

    std::vector<TestType> data(16);
    for(std::size_t i = 0; i < data.size(); ++i)
        data[i] = static_cast<TestType>(i);

    std::span<TestType> data_span(data.data(), data.size());

    using shape_type = scalar_type::shape_type;

    shape_type scalar_shape({});
    shape_type vector_shape({16});
    shape_type matrix_shape({4, 4});
    shape_type tensor3_shape({2, 2, 4});
    shape_type tensor4_shape({2, 2, 2, 2});

    scalar_type scalar(data_span, scalar_shape);
    vector_type vector(data_span, vector_shape);
    matrix_type matrix(data_span, matrix_shape);
    tensor3_type tensor3(data_span, tensor3_shape);
    tensor4_type tensor4(data_span, tensor4_shape);

    SECTION("permuted_copy") {
        using label_type = typename scalar_type::label_type;
        SECTION("scalar") {
            label_type label("");
            auto&& [scalar_buffer, pscalar] =
              scalar.permuted_copy(label, label);
            REQUIRE(scalar_buffer.size() == 1);
            REQUIRE(pscalar->get_elem({}) == data[0]);
            REQUIRE(scalar_buffer.data() != data.data()); // Ensure a copy
        }

        SECTION("vector") {
            label_type i("i");
            auto&& [vector_buffer, pvector] = vector.permuted_copy(i, i);
            REQUIRE(vector_buffer.size() == 16);
            for(std::size_t idx = 0; idx < 16; ++idx) {
                REQUIRE(pvector->get_elem({idx}) == data[idx]);
            }
            REQUIRE(vector_buffer.data() != data.data()); // Ensure a copy
        }

        SECTION("matrix") {
            label_type ij("i,j");
            label_type ji("j,i");
            SECTION("no permutation") {
                auto&& [matrix_buffer, pmatrix] = matrix.permuted_copy(ij, ij);
                REQUIRE(matrix_buffer.size() == 16);
                for(std::size_t idx = 0; idx < 4; ++idx) {
                    for(std::size_t jdx = 0; jdx < 4; ++jdx) {
                        REQUIRE(pmatrix->get_elem({idx, jdx}) ==
                                data[idx * 4 + jdx]);
                    }
                }
                REQUIRE(matrix_buffer.data() != data.data()); // Ensure a copy
            }
            SECTION("Permutation") {
                auto&& [matrix_buffer, pmatrix] = matrix.permuted_copy(ji, ij);
                REQUIRE(matrix_buffer.size() == 16);
                for(std::size_t idx = 0; idx < 4; ++idx) {
                    for(std::size_t jdx = 0; jdx < 4; ++jdx) {
                        REQUIRE(pmatrix->get_elem({jdx, idx}) ==
                                data[idx * 4 + jdx]);
                    }
                }
                REQUIRE(matrix_buffer.data() != data.data()); // Ensure a copy
            }
        }

        SECTION("Rank 3 tensor") {
            label_type ijk("i,j,k");
            label_type jik("j,i,k");
            SECTION("no permutation") {
                auto&& [t_buffer, pt] = tensor3.permuted_copy(ijk, ijk);
                REQUIRE(t_buffer.size() == 16);
                for(std::size_t idx = 0; idx < 2; ++idx) {
                    for(std::size_t jdx = 0; jdx < 2; ++jdx) {
                        for(std::size_t kdx = 0; kdx < 4; ++kdx) {
                            REQUIRE(pt->get_elem({idx, jdx, kdx}) ==
                                    data[idx * 8 + jdx * 4 + kdx]);
                        }
                    }
                }
                REQUIRE(t_buffer.data() != data.data()); // Ensure a copy
            }
            SECTION("Permutation") {
                auto&& [t_buffer, pt] = tensor3.permuted_copy(jik, ijk);
                REQUIRE(t_buffer.size() == 16);
                for(std::size_t idx = 0; idx < 2; ++idx) {
                    for(std::size_t jdx = 0; jdx < 2; ++jdx) {
                        for(std::size_t kdx = 0; kdx < 4; ++kdx) {
                            REQUIRE(pt->get_elem({jdx, idx, kdx}) ==
                                    data[idx * 8 + jdx * 4 + kdx]);
                        }
                    }
                }
                REQUIRE(t_buffer.data() != data.data()); // Ensure a copy
            }
        }

        SECTION("Rank 4 tensor") {
            label_type ijkl("i,j,k,l");
            label_type jikl("j,i,k,l");
            SECTION("no permutation") {
                auto&& [t_buffer, pt] = tensor4.permuted_copy(ijkl, ijkl);
                REQUIRE(t_buffer.size() == 16);
                for(std::size_t idx = 0; idx < 2; ++idx) {
                    for(std::size_t jdx = 0; jdx < 2; ++jdx) {
                        for(std::size_t kdx = 0; kdx < 2; ++kdx) {
                            for(std::size_t ldx = 0; ldx < 2; ++ldx) {
                                REQUIRE(
                                  pt->get_elem({idx, jdx, kdx, ldx}) ==
                                  data[idx * 8 + jdx * 4 + kdx * 2 + ldx]);
                            }
                        }
                    }
                }
                REQUIRE(t_buffer.data() != data.data()); // Ensure a copy
            }
            SECTION("Permutation") {
                auto&& [t_buffer, pt] = tensor4.permuted_copy(jikl, ijkl);
                REQUIRE(t_buffer.size() == 16);
                for(std::size_t idx = 0; idx < 2; ++idx) {
                    for(std::size_t jdx = 0; jdx < 2; ++jdx) {
                        for(std::size_t kdx = 0; kdx < 2; ++kdx) {
                            for(std::size_t ldx = 0; ldx < 2; ++ldx) {
                                REQUIRE(
                                  pt->get_elem({jdx, idx, kdx, ldx}) ==
                                  data[idx * 8 + jdx * 4 + kdx * 2 + ldx]);
                            }
                        }
                    }
                }
                REQUIRE(t_buffer.data() != data.data()); // Ensure a copy
            }
        }
    }

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

    SECTION("extent") {
        REQUIRE(vector.extent(0) == 16);

        REQUIRE(matrix.extent(0) == 4);
        REQUIRE(matrix.extent(1) == 4);

        REQUIRE(tensor3.extent(0) == 2);
        REQUIRE(tensor3.extent(1) == 2);
        REQUIRE(tensor3.extent(2) == 4);

        REQUIRE(tensor4.extent(0) == 2);
        REQUIRE(tensor4.extent(1) == 2);
        REQUIRE(tensor4.extent(2) == 2);
        REQUIRE(tensor4.extent(3) == 2);
    }

    SECTION("get_elem") {
        REQUIRE(scalar.get_elem({}) == data[0]);

        REQUIRE(vector.get_elem({0}) == data[0]);
        REQUIRE(vector.get_elem({15}) == data[15]);

        REQUIRE(matrix.get_elem({0, 0}) == data[0]);
        REQUIRE(matrix.get_elem({3, 3}) == data[15]);

        REQUIRE(tensor3.get_elem({0, 0, 0}) == data[0]);
        REQUIRE(tensor3.get_elem({1, 1, 3}) == data[15]);

        REQUIRE(tensor4.get_elem({0, 0, 0, 0}) == data[0]);
        REQUIRE(tensor4.get_elem({1, 1, 1, 1}) == data[15]);
    }

    SECTION("set_elem") {
        TestType corr(42);
        scalar.set_elem({}, 42);
        REQUIRE(scalar.get_elem({}) == corr);

        vector.set_elem({5}, 42);
        REQUIRE(vector.get_elem({5}) == corr);

        matrix.set_elem({2, 2}, 42);
        REQUIRE(matrix.get_elem({2, 2}) == corr);

        tensor3.set_elem({1, 0, 3}, 42);
        REQUIRE(tensor3.get_elem({1, 0, 3}) == corr);

        tensor4.set_elem({0, 1, 1, 0}, 42);
        REQUIRE(tensor4.get_elem({0, 1, 1, 0}) == corr);
    }

    SECTION("data()") {
        REQUIRE(scalar.data().data() == data.data());
        REQUIRE(vector.data().data() == data.data());
        REQUIRE(matrix.data().data() == data.data());
        REQUIRE(tensor3.data().data() == data.data());
        REQUIRE(tensor4.data().data() == data.data());
    }

    SECTION("data() const") {
        REQUIRE(std::as_const(scalar).data().data() == data.data());
        REQUIRE(std::as_const(vector).data().data() == data.data());
        REQUIRE(std::as_const(matrix).data().data() == data.data());
        REQUIRE(std::as_const(tensor3).data().data() == data.data());
        REQUIRE(std::as_const(tensor4).data().data() == data.data());
    }

    SECTION("fill") {
        TestType corr(7);
        SECTION("scalar") {
            scalar.fill(corr);
            REQUIRE(scalar.get_elem({}) == corr);
        }

        SECTION("vector") {
            vector.fill(corr);
            for(std::size_t i = 0; i < vector.size(); ++i)
                REQUIRE(vector.get_elem({i}) == corr);
        }

        SECTION("matrix") {
            matrix.fill(corr);
            for(std::size_t i = 0; i < matrix.extent(0); ++i)
                for(std::size_t j = 0; j < matrix.extent(1); ++j)
                    REQUIRE(matrix.get_elem({i, j}) == corr);
        }

        SECTION("rank 3 tensor") {
            tensor3.fill(corr);
            for(std::size_t i = 0; i < tensor3.extent(0); ++i)
                for(std::size_t j = 0; j < tensor3.extent(1); ++j)
                    for(std::size_t k = 0; k < tensor3.extent(2); ++k)
                        REQUIRE(tensor3.get_elem({i, j, k}) == corr);
        }

        SECTION("rank 4 tensor") {
            tensor4.fill(corr);
            for(std::size_t i = 0; i < tensor4.extent(0); ++i)
                for(std::size_t j = 0; j < tensor4.extent(1); ++j)
                    for(std::size_t k = 0; k < tensor4.extent(2); ++k)
                        for(std::size_t l = 0; l < tensor4.extent(3); ++l)
                            REQUIRE(tensor4.get_elem({i, j, k, l}) == corr);
        }
    }

    SECTION("addition_assignment") {
        SECTION("scalar") {
            testing::scalar_addition_assignment<scalar_type>();
        }

        SECTION("vector") {
            testing::vector_addition_assignment<vector_type>();
        }

        SECTION("matrix") {
            testing::matrix_addition_assignment<matrix_type>();
        }

        SECTION("rank 3 tensor") {
            testing::tensor3_addition_assignment<tensor3_type>();
        }

        SECTION("rank 4 tensor") {
            testing::tensor4_addition_assignment<tensor4_type>();
        }
    }

    SECTION("subtraction_assignment") {
        SECTION("scalar") {
            testing::scalar_subtraction_assignment<scalar_type>();
        }

        SECTION("vector") {
            testing::vector_subtraction_assignment<vector_type>();
        }

        SECTION("matrix") {
            testing::matrix_subtraction_assignment<matrix_type>();
        }

        SECTION("rank 3 tensor") {
            testing::tensor3_subtraction_assignment<tensor3_type>();
        }

        SECTION("rank 4 tensor") {
            testing::tensor4_subtraction_assignment<tensor4_type>();
        }
    }

    SECTION("hadamard_assignment") {
        SECTION("scalar") {
            testing::scalar_hadamard_assignment<scalar_type>();
        }

        SECTION("vector") {
            testing::vector_hadamard_assignment<vector_type>();
        }

        SECTION("matrix") {
            testing::matrix_hadamard_assignment<matrix_type>();
        }

        SECTION("rank 3 tensor") {
            testing::tensor3_hadamard_assignment<tensor3_type>();
        }

        SECTION("rank 4 tensor") {
            testing::tensor4_hadamard_assignment<tensor4_type>();
        }
    }

    SECTION("permute_assignment") {
        SECTION("scalar") { testing::scalar_permute_assignment<scalar_type>(); }
        SECTION("vector") { testing::vector_permute_assignment<vector_type>(); }
        SECTION("matrix") { testing::matrix_permute_assignment<matrix_type>(); }
        SECTION("rank 3 tensor") {
            testing::tensor3_permute_assignment<tensor3_type>();
        }
        SECTION("rank 4 tensor") {
            testing::tensor4_permute_assignment<tensor4_type>();
        }
    }

    SECTION("scalar_multiplication") {
        SECTION("scalar") {
            testing::scalar_scalar_multiplication<scalar_type>();
        }
        SECTION("vector") {
            testing::vector_scalar_multiplication<vector_type>();
        }
        SECTION("matrix") {
            testing::matrix_scalar_multiplication<matrix_type>();
        }
        SECTION("rank 3 tensor") {
            testing::tensor3_scalar_multiplication<tensor3_type>();
        }
        SECTION("rank 4 tensor") {
            testing::tensor4_scalar_multiplication<tensor4_type>();
        }
    }

    SECTION("contraction_assignment") {
        testing::contraction_assignment_tests<
          scalar_type, vector_type, matrix_type, tensor3_type, tensor4_type>();
    }
}
