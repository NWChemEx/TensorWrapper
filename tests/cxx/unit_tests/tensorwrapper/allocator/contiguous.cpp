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
#include <tensorwrapper/allocator/contiguous.hpp>

using namespace tensorwrapper;

TEMPLATE_LIST_TEST_CASE("allocator::Contiguous", "",
                        types::floating_point_types) {
    using allocator_type = allocator::Contiguous<TestType>;
    using layout_type    = typename allocator_type::layout_type;

    auto alloc = testing::make_allocator<TestType>();

    auto scalar_corr = testing::eigen_scalar<TestType>();
    auto vector_corr = testing::eigen_vector<TestType>();
    auto matrix_corr = testing::eigen_matrix<TestType>();

    SECTION("allocate(layout)") {
        auto pscalar  = alloc.allocate(scalar_corr->layout());
        pscalar->at() = 42.0;
        REQUIRE(pscalar->are_equal(*scalar_corr));
    }

    SECTION("allocate(layout*)") {
        auto pvector   = alloc.allocate(vector_corr->layout());
        pvector->at(0) = 0.0;
        pvector->at(1) = 1.0;
        pvector->at(2) = 2.0;
        pvector->at(3) = 3.0;
        pvector->at(4) = 4.0;

        REQUIRE(pvector->are_equal(*vector_corr));
    }

    SECTION("contruct(scalar)") {
        auto pscalar = alloc.construct(42.0);
        REQUIRE(pscalar->are_equal(*scalar_corr));
    }

    SECTION("construct(vector)") {
        auto pvector = alloc.construct({0.0, 1.0, 2.0, 3.0, 4.0});
        REQUIRE(pvector->are_equal(*vector_corr));
    }

    SECTION("construct(matrix)") {
        typename allocator_type::rank2_il il{{1.0, 2.0}, {3.0, 4.0}};
        auto pmatrix = alloc.construct(il);
        REQUIRE(pmatrix->are_equal(*matrix_corr));
    }

    SECTION("construct(tensor3)") {
        typename allocator_type::rank3_il il{{{1.0, 2.0}, {3.0, 4.0}},
                                             {{5.0, 6.0}, {7.0, 8.0}}};
        auto ptensor3 = alloc.construct(il);
        REQUIRE(ptensor3->are_equal(*testing::eigen_tensor3<TestType>()));
    }

    SECTION("construct(tensor4)") {
        typename allocator_type::rank4_il il{
          {{{1.0, 2.0}, {3.0, 4.0}}, {{5.0, 6.0}, {7.0, 8.0}}},
          {{{9.0, 10.0}, {11.0, 12.0}}, {{13.0, 14.0}, {15.0, 16.0}}}};
        auto ptensor4 = alloc.construct(il);
        REQUIRE(ptensor4->are_equal(*testing::eigen_tensor4<TestType>()));
    }

    SECTION("construct(layout, value)") {
        auto pmatrix          = alloc.construct(matrix_corr->layout(), 0.0);
        matrix_corr->at(0, 0) = 0.0;
        matrix_corr->at(0, 1) = 0.0;
        matrix_corr->at(1, 0) = 0.0;
        matrix_corr->at(1, 1) = 0.0;

        REQUIRE(pmatrix->are_equal(*matrix_corr));
    }

    SECTION("construct(layout*, value)") {
        auto pmatrix = alloc.construct(
          matrix_corr->layout().template clone_as<layout_type>(), 0.0);
        matrix_corr->at(0, 0) = 0.0;
        matrix_corr->at(0, 1) = 0.0;
        matrix_corr->at(1, 0) = 0.0;
        matrix_corr->at(1, 1) = 0.0;

        REQUIRE(pmatrix->are_equal(*matrix_corr));
    }
}