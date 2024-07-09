#include <catch2/catch.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/layout/mono_tile.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace buffer;

/* Testing strategy:
 *
 * - BufferBase is an abstract class. To test it we must create an instance of
 *   a derived class. We then will upcast to BufferBase and perform checks
 *   through the BufferBase interface.
 *
 */

TEST_CASE("BufferBase") {
    if constexpr(have_eigen()) {
        using scalar_buffer = buffer::Eigen<float, 0>;
        using vector_buffer = buffer::Eigen<float, 1>;

        typename scalar_buffer::tensor_type eigen_scalar;
        eigen_scalar() = 1.0;

        typename vector_buffer::tensor_type eigen_vector(2);
        eigen_vector(0) = 1.0;
        eigen_vector(1) = 2.0;

        symmetry::Group g;
        sparsity::Pattern p;
        layout::MonoTile scalar_layout(shape::Smooth{}, g, p);
        layout::MonoTile vector_layout(shape::Smooth{2}, g, p);

        vector_buffer defaulted;
        scalar_buffer scalar(eigen_scalar, scalar_layout);
        vector_buffer vector(eigen_vector, vector_layout);

        BufferBase& defaulted_base = defaulted;
        BufferBase& scalar_base    = scalar;
        BufferBase& vector_base    = vector;

        SECTION("has_layout") {
            REQUIRE_FALSE(defaulted_base.has_layout());
            REQUIRE(scalar_base.has_layout());
            REQUIRE(vector_base.has_layout());
        }

        SECTION("layout") {
            REQUIRE_THROWS_AS(defaulted_base.layout(), std::runtime_error);
            REQUIRE(scalar_base.layout().are_equal(scalar_layout));
            REQUIRE(vector_base.layout().are_equal(vector_layout));
        }

        SECTION("operator==") {
            // Defaulted layout == defaulted layout
            REQUIRE(defaulted_base == scalar_buffer());

            // Defaulted layout != non-defaulted layout
            REQUIRE_FALSE(defaulted_base == scalar_base);

            // Non-defaulted layout same value
            REQUIRE(scalar_base == scalar_buffer(eigen_scalar, scalar_layout));

            // Non-defaulted layout different value
            REQUIRE_FALSE(scalar_base == vector_base);
        }

        SECTION("operator!=") {
            // Just spot check because it negates operator==, which was tested
            REQUIRE(defaulted_base != scalar_base);
            REQUIRE_FALSE(defaulted_base != scalar_buffer());
        }
    }
}
