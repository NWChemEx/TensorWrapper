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

TEMPLATE_LIST_TEST_CASE("Eigen", "", types::floating_point_types) {
    // N.B. we actually get Contiguous<TestType> objects back
    using buffer_type = buffer::Eigen<TestType>;

    auto pscalar       = testing::eigen_scalar<TestType>();
    auto& eigen_scalar = static_cast<buffer_type&>(*pscalar);
    eigen_scalar.at()  = 10.0;

    auto pvector       = testing::eigen_vector<TestType>(2);
    auto& eigen_vector = static_cast<buffer_type&>(*pvector);
    eigen_vector.at(0) = 10.0;
    eigen_vector.at(1) = 20.0;

    auto pmatrix          = testing::eigen_matrix<TestType>(2, 3);
    auto& eigen_matrix    = static_cast<buffer_type&>(*pmatrix);
    eigen_matrix.at(0, 0) = 10.0;
    eigen_matrix.at(0, 1) = 20.0;
    eigen_matrix.at(0, 2) = 30.0;
    eigen_matrix.at(1, 0) = 40.0;
    eigen_matrix.at(1, 1) = 50.0;
    eigen_matrix.at(1, 2) = 60.0;

    auto ptensor             = testing::eigen_tensor3<TestType>(1, 2, 3);
    auto& eigen_tensor       = static_cast<buffer_type&>(*ptensor);
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

    buffer_type defaulted;

    SECTION("ctors, assignment") {
        SECTION("default ctor") { REQUIRE(defaulted.data() == nullptr); }

        SECTION("value ctor") {
            REQUIRE(eigen_scalar.layout().are_equal(scalar_layout));
            REQUIRE(eigen_vector.layout().are_equal(vector_layout));
            REQUIRE(eigen_matrix.layout().are_equal(matrix_layout));
            REQUIRE(eigen_tensor.layout().are_equal(tensor_layout));
        }

        test_copy_move_ctor_and_assignment(eigen_scalar, eigen_vector,
                                           eigen_matrix, eigen_tensor);
    }

    SECTION("swap") {
        buffer_type copy(eigen_scalar);
        eigen_scalar.swap(defaulted);
        REQUIRE(defaulted == copy);
        REQUIRE(eigen_scalar == buffer_type{});
    }

    SECTION("operator==") {
        // Checking Layout/Allocator falls to base class tests
        auto pscalar2       = testing::eigen_scalar<TestType>();
        auto& eigen_scalar2 = static_cast<buffer_type&>(*pscalar2);
        eigen_scalar2.at()  = 10.0;

        // Defaulted != scalar
        REQUIRE_FALSE(defaulted == eigen_scalar);

        // Everything the same
        REQUIRE(eigen_scalar == eigen_scalar2);

        SECTION("Different buffer value") {
            eigen_scalar2.at() = 2.0;
            REQUIRE_FALSE(eigen_scalar == eigen_scalar2);
        }
    }

    SECTION("operator!=") {
        auto pscalar2       = testing::eigen_scalar<TestType>();
        auto& eigen_scalar2 = static_cast<buffer_type&>(*pscalar2);
        eigen_scalar2.at()  = 10.0;

        REQUIRE_FALSE(eigen_scalar != eigen_scalar2);
        eigen_scalar2.at() = 2.0;
        REQUIRE(eigen_scalar != eigen_scalar2);
    }

    SECTION("virtual method overrides") {
        SECTION("clone") {
            REQUIRE(eigen_scalar.clone()->are_equal(eigen_scalar));
            REQUIRE(eigen_vector.clone()->are_equal(eigen_vector));
            REQUIRE(eigen_matrix.clone()->are_equal(eigen_matrix));
        }

        SECTION("are_equal") {
            REQUIRE(eigen_scalar.are_equal(eigen_scalar));
            REQUIRE_FALSE(eigen_matrix.are_equal(eigen_scalar));
        }

        SECTION("addition_assignment") {
            buffer_type output;
            auto vi = eigen_vector("i");
            output.addition_assignment("i", vi, vi);

            auto corr   = testing::eigen_vector<TestType>(2);
            corr->at(0) = 20.0;
            corr->at(1) = 40.0;

            REQUIRE(output.are_equal(*corr));
        }

        SECTION("subtraction_assignment") {
            buffer_type output;
            auto vi = eigen_vector("i");
            output.subtraction_assignment("i", vi, vi);

            auto corr   = testing::eigen_vector<TestType>(2);
            corr->at(0) = 0.0;
            corr->at(1) = 0.0;

            REQUIRE(output.are_equal(*corr));
        }

        SECTION("multiplication_assignment") {
            buffer_type output;
            auto vi = eigen_vector("i");
            output.multiplication_assignment("i", vi, vi);

            auto corr   = testing::eigen_vector<TestType>(2);
            corr->at(0) = 100.0;
            corr->at(1) = 400.0;

            REQUIRE(output.are_equal(*corr));
        }

        SECTION("permute_assignment") {
            buffer_type output;
            auto mij = eigen_matrix("i,j");
            output.permute_assignment("j,i", mij);

            auto corr      = testing::eigen_matrix<TestType>(3, 2);
            corr->at(0, 0) = 10.0;
            corr->at(0, 1) = 40.0;
            corr->at(1, 0) = 20.0;
            corr->at(1, 1) = 50.0;
            corr->at(2, 0) = 30.0;
            corr->at(2, 1) = 60.0;

            REQUIRE(output.are_equal(*corr));
        }

        SECTION("scalar_multiplication") {
            buffer_type output;
            auto vi = eigen_vector("i");
            output.scalar_multiplication("i", 2.0, vi);

            auto corr   = testing::eigen_vector<TestType>(2);
            corr->at(0) = 20.0;
            corr->at(1) = 40.0;

            REQUIRE(output.are_equal(*corr));
        }

        SECTION("data()") {
            REQUIRE(defaulted.data() == nullptr);
            REQUIRE(*eigen_scalar.data() == TestType{10.0});
            REQUIRE(*eigen_matrix.data() == TestType{10.0});
        }

        SECTION("data() const") {
            REQUIRE(std::as_const(defaulted).data() == nullptr);
            REQUIRE(*std::as_const(eigen_scalar).data() == TestType{10.0});
            REQUIRE(*std::as_const(eigen_matrix).data() == TestType{10.0});
        }

        SECTION("get_elem_()") {
            REQUIRE(eigen_scalar.at() == TestType{10.0});
            REQUIRE(eigen_vector.at(0) == TestType{10.0});
            REQUIRE(eigen_matrix.at(0, 0) == TestType{10.0});
        }

        SECTION("get_elem_() const") {
            REQUIRE(std::as_const(eigen_scalar).at() == TestType{10.0});
            REQUIRE(std::as_const(eigen_vector).at(0) == TestType{10.0});
            REQUIRE(std::as_const(eigen_matrix).at(0, 0) == TestType{10.0});
        }
    }
}
