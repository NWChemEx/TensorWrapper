#include "../helpers.hpp"
#include <parallelzone/parallelzone.hpp>
#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;

TEMPLATE_TEST_CASE("EigenAllocator", "", float, double) {
    using scalar_alloc_type   = allocator::Eigen<TestType, 0>;
    using vector_alloc_type   = allocator::Eigen<TestType, 1>;
    using matrix_alloc_type   = allocator::Eigen<TestType, 2>;
    using layout_type         = typename scalar_alloc_type::eigen_layout_type;
    using shape_type          = typename shape::Smooth;
    using symmetry_type       = typename layout_type::symmetry_type;
    using sparsity_type       = typename layout_type::sparsity_type;
    using eigen_buffer_scalar = typename scalar_alloc_type::eigen_buffer_type;
    using eigen_buffer_vector = typename vector_alloc_type::eigen_buffer_type;
    using eigen_buffer_matrix = typename matrix_alloc_type::eigen_buffer_type;
    using eigen_scalar        = typename eigen_buffer_scalar::tensor_type;
    using eigen_vector        = typename eigen_buffer_vector::tensor_type;
    using eigen_matrix        = typename eigen_buffer_matrix::tensor_type;

    parallelzone::runtime::RuntimeView rv;

    symmetry_type g;
    sparsity_type sparsity;
    layout_type scalar_layout(shape_type{}, g, sparsity);
    layout_type vector_layout(shape_type{2}, g, sparsity);
    layout_type matrix_layout(shape_type{2, 2}, g, sparsity);

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
