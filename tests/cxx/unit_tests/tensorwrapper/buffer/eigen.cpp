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

#include "../testing/testing.hpp"
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/layout/physical.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace testing;

TEMPLATE_LIST_TEST_CASE("Eigen", "", testing::floating_point_types) {
    using buffer_type = buffer::Eigen<TestType>;

    auto pscalar       = testing::eigen_scalar<TestType>();
    auto& eigen_scalar = *pscalar;
    eigen_scalar.at()  = 10.0;

    auto pvector       = testing::eigen_vector<TestType>(2);
    auto& eigen_vector = *pvector;
    eigen_vector.at(0) = 10.0;
    eigen_vector.at(1) = 20.0;

    auto pmatrix          = testing::eigen_matrix<TestType>(2, 3);
    auto& eigen_matrix    = *pmatrix;
    eigen_matrix.at(0, 0) = 10.0;
    eigen_matrix.at(0, 1) = 20.0;
    eigen_matrix.at(0, 2) = 30.0;
    eigen_matrix.at(1, 0) = 40.0;
    eigen_matrix.at(1, 1) = 50.0;
    eigen_matrix.at(1, 2) = 60.0;

    auto ptensor             = testing::eigen_tensor3<TestType>(1, 2, 3);
    auto& eigen_tensor       = *ptensor;
    eigen_tensor.at(0, 0, 0) = 10.0;
    eigen_tensor.at(0, 0, 1) = 20.0;
    eigen_tensor.at(0, 0, 2) = 30.0;
    eigen_tensor.at(0, 1, 0) = 40.0;
    eigen_tensor.at(0, 1, 1) = 50.0;
    eigen_tensor.at(0, 1, 2) = 60.0;

    auto scalar_layout = scalar_physical();
    auto vector_layout = vector_physical(2);
    auto matrix_layout = matrix_physical(2, 3);
    auto tensor_layout = tensor3_physical(1, 2, 3);

    SECTION("ctors, assignment") {
        SECTION("value ctor") {
            REQUIRE(eigen_scalar.layout().are_equal(scalar_layout));
            REQUIRE(eigen_vector.layout().are_equal(vector_layout));
            REQUIRE(eigen_matrix.layout().are_equal(matrix_layout));
            REQUIRE(eigen_tensor.layout().are_equal(tensor_layout));
        }

        // test_copy_move_ctor_and_assignment(eigen_scalar, eigen_vector,
        //                                    eigen_matrix, eigen_tensor);
    }

    SECTION("operator==") {
        // We assume the eigen tensor and the layout objects work. So when
        // comparing Eigen objects we have four states: same everything,
        // different everything, same tensor different layout, and different
        // tensor same layout.

        auto pscalar2       = testing::eigen_scalar<TestType>();
        auto& eigen_scalar2 = *pscalar2;
        eigen_scalar2.at()  = 10.0;

        // Everything the same
        REQUIRE(eigen_scalar == eigen_scalar2);

        // SECTION("Different scalar") {
        //     eigen_scalar2.at() = 2.0;
        //     REQUIRE_FALSE(eigen_scalar == eigen_scalar2);
        // }
    }

    SECTION("operator!=") {
        // This just negates operator== so spot-checking is okay

        auto pscalar2       = testing::eigen_scalar<TestType>();
        auto& eigen_scalar2 = *pscalar2;
        eigen_scalar2.at()  = 10.0;

        // Everything the same
        REQUIRE_FALSE(eigen_scalar != eigen_scalar2);

        // eigen_scalar2.at() = 2.0;
        // REQUIRE(eigen_scalar2 != eigen_scalar);
    }

    SECTION("virtual method overrides") {
        using const_reference =
          typename buffer_type::const_buffer_base_reference;
        const_reference pscalar = eigen_scalar;
        const_reference pvector = eigen_vector;
        const_reference pmatrix = eigen_matrix;

        SECTION("clone") {
            REQUIRE(pscalar.clone()->are_equal(pscalar));
            REQUIRE(pvector.clone()->are_equal(pvector));
            REQUIRE(pmatrix.clone()->are_equal(pmatrix));
        }

        SECTION("are_equal") {
            REQUIRE(pscalar.are_equal(eigen_scalar));
            REQUIRE_FALSE(pmatrix.are_equal(eigen_scalar));
        }
    }

    SECTION("multiplication_assignment_") {
        // Multiplication just dispatches to hadamard_ or contraction_
        // Here we test the error-handling

        // Must be either a pure hadamard or a pure contraction
        auto matrix2 = testing::eigen_matrix<TestType>();
        auto mij     = eigen_matrix("i,j");

        REQUIRE_THROWS_AS(matrix2->subtraction_assignment("i", mij, mij),
                          std::runtime_error);
    }
}
