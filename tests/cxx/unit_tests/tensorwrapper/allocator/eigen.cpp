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

TEMPLATE_TEST_CASE("EigenAllocator", "", float, double) {
    using scalar_alloc_type   = allocator::Eigen<TestType, 0>;
    using vector_alloc_type   = allocator::Eigen<TestType, 1>;
    using matrix_alloc_type   = allocator::Eigen<TestType, 2>;
    using eigen_buffer_scalar = typename scalar_alloc_type::eigen_buffer_type;
    using eigen_buffer_vector = typename vector_alloc_type::eigen_buffer_type;
    using eigen_buffer_matrix = typename matrix_alloc_type::eigen_buffer_type;
    using eigen_scalar        = typename eigen_buffer_scalar::data_type;
    using eigen_vector        = typename eigen_buffer_vector::data_type;
    using eigen_matrix        = typename eigen_buffer_matrix::data_type;

    parallelzone::runtime::RuntimeView rv;

    auto scalar_layout = testing::scalar_physical();
    auto vector_layout = testing::vector_physical(2);
    auto matrix_layout = testing::matrix_physical(2, 2);
    using layout_type  = decltype(scalar_layout);

    scalar_alloc_type scalar_alloc(rv);
    vector_alloc_type vector_alloc(rv);
    matrix_alloc_type matrix_alloc(rv);

    eigen_scalar scalar;
    scalar() = 0.0;
    eigen_buffer_scalar scalar_corr(scalar, scalar_layout);

    eigen_vector vector(2);
    vector.setConstant(1);
    eigen_buffer_vector vector_corr(vector, vector_layout);

    eigen_matrix matrix(2, 2);
    matrix.setConstant(2);
    eigen_buffer_matrix matrix_corr(matrix, matrix_layout);

    SECTION("Ctor") {
        SECTION("runtime") {
            REQUIRE(scalar_alloc.runtime() == rv);
            REQUIRE(vector_alloc.runtime() == rv);
            REQUIRE(matrix_alloc.runtime() == rv);
        }

        testing::test_copy_and_move_ctors(scalar_alloc, vector_alloc,
                                          matrix_alloc);
    }

    SECTION("allocate(MonoTile)") {
        // N.b. allocate doesn't initialize tensor, so only compare layouts
        auto pscalar = scalar_alloc.allocate(scalar_layout);
        REQUIRE(pscalar->layout().are_equal(scalar_layout));

        auto pvector = vector_alloc.allocate(vector_layout);
        REQUIRE(pvector->layout().are_equal(vector_layout));

        auto pmatrix = matrix_alloc.allocate(matrix_layout);
        REQUIRE(pmatrix->layout().are_equal(matrix_layout));

        // Throws if ranks don't match
        using except_t = std::runtime_error;
        REQUIRE_THROWS_AS(scalar_alloc.allocate(vector_layout), except_t);
    }

    SECTION("allocate(std::unique_ptr<MonoTile>)") {
        // N.b. allocate doesn't initialize tensor, so only compare layouts
        auto pscalar_layout = std::make_unique<layout_type>(scalar_layout);
        auto pscalar        = scalar_alloc.allocate(std::move(pscalar_layout));
        REQUIRE(pscalar->layout().are_equal(scalar_layout));

        auto pvector_layout = std::make_unique<layout_type>(vector_layout);
        auto pvector        = vector_alloc.allocate(std::move(pvector_layout));
        REQUIRE(pvector->layout().are_equal(vector_layout));

        auto pmatrix_layout = std::make_unique<layout_type>(matrix_layout);
        auto pmatrix        = matrix_alloc.allocate(std::move(pmatrix_layout));
        REQUIRE(pmatrix->layout().are_equal(matrix_layout));

        // Throws if ranks don't match
        using except_t       = std::runtime_error;
        auto pvector_layout2 = std::make_unique<layout_type>(vector_layout);
        REQUIRE_THROWS_AS(scalar_alloc.allocate(std::move(pvector_layout2)),
                          except_t);
    }

    SECTION("construct(value)") {
        auto pscalar = scalar_alloc.construct(scalar_layout, 0);
        REQUIRE(*pscalar == scalar_corr);

        auto pvector = vector_alloc.construct(vector_layout, 1);
        REQUIRE(*pvector == vector_corr);

        auto pmatrix_layout = std::make_unique<layout_type>(matrix_layout);
        auto pmatrix = matrix_alloc.construct(std::move(pmatrix_layout), 2);
        REQUIRE(*pmatrix == matrix_corr);

        // Throws if ranks don't match
        using except_t = std::runtime_error;
        REQUIRE_THROWS_AS(scalar_alloc.allocate(vector_layout), except_t);
    }

    SECTION("can_rebind") {
        REQUIRE(scalar_alloc.can_rebind(scalar_corr));
        REQUIRE_FALSE(scalar_alloc.can_rebind(vector_corr));
    }

    SECTION("rebind(non-const)") {
        using type         = typename scalar_alloc_type::buffer_base_reference;
        type scalar_base   = scalar_corr;
        auto& eigen_buffer = scalar_alloc.rebind(scalar_base);
        REQUIRE(&eigen_buffer == &scalar_corr);
        REQUIRE_THROWS_AS(scalar_alloc.rebind(vector_corr), std::runtime_error);
    }

    SECTION("rebind(const)") {
        using type = typename scalar_alloc_type::const_buffer_base_reference;
        type scalar_base   = scalar_corr;
        auto& eigen_buffer = scalar_alloc.rebind(scalar_base);
        REQUIRE(&eigen_buffer == &scalar_corr);

        type vector_base = vector_corr;
        REQUIRE_THROWS_AS(scalar_alloc.rebind(vector_base), std::runtime_error);
    }

    SECTION("operator==") {
        REQUIRE(scalar_alloc == scalar_alloc_type(rv));
        REQUIRE_FALSE(scalar_alloc == vector_alloc);
    }

    SECTION("virtual_methods") {
        SECTION("clone") {
            auto pscalar = scalar_alloc.clone();
            REQUIRE(pscalar->are_equal(scalar_alloc));
        }

        SECTION("are_equal") {
            REQUIRE(scalar_alloc.are_equal(scalar_alloc_type(rv)));
            REQUIRE_FALSE(scalar_alloc.are_equal(vector_alloc));
        }
    }
}
