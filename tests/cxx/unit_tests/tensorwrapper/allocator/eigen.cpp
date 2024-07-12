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
    using eigen_scalar        = typename eigen_buffer_scalar::tensor_type;
    // using eigen_vector      = typename vector_alloc_type::eigen_buffer_type;
    // using eigen_matrix      = typename matrix_alloc_type::eigen_buffer_type;

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

    SECTION("Ctor") {
        SECTION("runtime") {
            REQUIRE(scalar_alloc.runtime() == rv);
            REQUIRE(vector_alloc.runtime() == rv);
            REQUIRE(matrix_alloc.runtime() == rv);
        }
    }

    SECTION("allocate(MonoTile)") {
        // N.b. allocate doesn't initialize tensor, so only compare layouts
        auto pscalar = scalar_alloc.allocate(scalar_layout);
        REQUIRE(pscalar->layout().are_equal(scalar_layout));
    }
}
