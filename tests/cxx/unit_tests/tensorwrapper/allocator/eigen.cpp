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
#include <parallelzone/parallelzone.hpp>
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;

using types2test = types::floating_point_types;

TEMPLATE_LIST_TEST_CASE("EigenAllocator", "", types2test) {
    using alloc_type = allocator::Eigen<TestType>;

    parallelzone::runtime::RuntimeView rv;
    auto scalar_layout = testing::scalar_physical();
    auto vector_layout = testing::vector_physical(2);
    auto matrix_layout = testing::matrix_physical(2, 2);
    using layout_type  = decltype(scalar_layout);

    auto pscalar_corr = testing::eigen_scalar<TestType>();
    auto& scalar_corr = *pscalar_corr;
    scalar_corr.at()  = 0.0;

    auto pvector_corr = testing::eigen_vector<TestType>(2);
    auto& vector_corr = *pvector_corr;
    vector_corr.at(0) = 1;
    vector_corr.at(1) = 1;

    auto pmatrix_corr    = testing::eigen_matrix<TestType>(2, 2);
    auto& matrix_corr    = *pmatrix_corr;
    matrix_corr.at(0, 0) = 2;
    matrix_corr.at(0, 1) = 2;
    matrix_corr.at(1, 0) = 2;
    matrix_corr.at(1, 1) = 2;

    alloc_type alloc(rv);

    SECTION("Ctor") {
        SECTION("runtime") { REQUIRE(alloc.runtime() == rv); }
        testing::test_copy_and_move_ctors(alloc);
    }

    SECTION("allocate(Layout)") {
        // N.b. allocate doesn't initialize tensor, so only compare layouts
        auto pscalar = alloc.allocate(scalar_layout);
        REQUIRE(pscalar->layout().are_equal(scalar_layout));

        auto pvector = alloc.allocate(vector_layout);
        REQUIRE(pvector->layout().are_equal(vector_layout));

        auto pmatrix = alloc.allocate(matrix_layout);
        REQUIRE(pmatrix->layout().are_equal(matrix_layout));

        // Works if ranks don't match
        pvector = alloc.allocate(vector_layout);
        REQUIRE(pvector->layout().are_equal(vector_layout));
    }

    SECTION("allocate(std::unique_ptr<Layout>)") {
        // N.b. allocate doesn't initialize tensor, so only compare layouts
        auto pscalar_layout = std::make_unique<layout_type>(scalar_layout);
        auto pscalar        = alloc.allocate(std::move(pscalar_layout));
        REQUIRE(pscalar->layout().are_equal(scalar_layout));

        auto pvector_layout = std::make_unique<layout_type>(vector_layout);
        auto pvector        = alloc.allocate(std::move(pvector_layout));
        REQUIRE(pvector->layout().are_equal(vector_layout));

        auto pmatrix_layout = std::make_unique<layout_type>(matrix_layout);
        auto pmatrix        = alloc.allocate(std::move(pmatrix_layout));
        REQUIRE(pmatrix->layout().are_equal(matrix_layout));
    }

    SECTION("construct(value)") {
        auto pscalar = alloc.construct(scalar_layout, 0);
        REQUIRE(*pscalar == scalar_corr);

        auto pvector = alloc.construct(vector_layout, 1);
        REQUIRE(*pvector == vector_corr);

        auto pmatrix_layout = std::make_unique<layout_type>(matrix_layout);
        auto pmatrix        = alloc.construct(std::move(pmatrix_layout), 2);
        REQUIRE(*pmatrix == matrix_corr);
    }

    SECTION("can_rebind") { REQUIRE(alloc.can_rebind(scalar_corr)); }

    SECTION("rebind(non-const)") {
        using type         = typename alloc_type::buffer_base_reference;
        type scalar_base   = scalar_corr;
        auto& eigen_buffer = alloc.rebind(scalar_base);
        REQUIRE(&eigen_buffer == &scalar_corr);
    }

    SECTION("rebind(const)") {
        using type         = typename alloc_type::const_buffer_base_reference;
        type scalar_base   = scalar_corr;
        auto& eigen_buffer = alloc.rebind(scalar_base);
        REQUIRE(&eigen_buffer == &scalar_corr);
    }

    SECTION("operator==") { REQUIRE(alloc == alloc_type(rv)); }

    SECTION("virtual_methods") {
        SECTION("clone") {
            auto pscalar = alloc.clone();
            REQUIRE(pscalar->are_equal(alloc));
        }

        SECTION("are_equal") { REQUIRE(alloc.are_equal(alloc_type(rv))); }
    }
}
