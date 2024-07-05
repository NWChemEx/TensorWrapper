#include "../helpers.hpp"
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/layout/mono_tile.hpp>
#include <tensorwrapper/shape/smooth.hpp>

using namespace tensorwrapper;
using namespace testing;

namespace {

template<typename LHSType, typename RHSType>
bool compare_eigen(const LHSType& lhs, const RHSType& rhs) {
    auto d                     = lhs - rhs;
    Eigen::Tensor<double, 0> r = d.sum();

    return (r() == 0.0);
}

} // namespace

TEMPLATE_TEST_CASE("Eigen", "", double) {
    if constexpr(have_eigen()) {
        using buffer_type = buffer::Eigen<TestType>;
        using scalar_type = typename buffer_type::template tensor_type<0>;
        using vector_type = typename buffer_type::template tensor_type<1>;
        using matrix_type = typename buffer_type::template tensor_type<2>;

        scalar_type eigen_scalar;
        eigen_scalar() = 1.0;

        vector_type eigen_vector(2);
        eigen_vector(0) = 1.0;
        eigen_vector(1) = 2.0;

        matrix_type eigen_matrix(2, 3);
        eigen_matrix(0, 0) = 1.0;
        eigen_matrix(0, 1) = 2.0;
        eigen_matrix(0, 2) = 3.0;
        eigen_matrix(1, 0) = 4.0;
        eigen_matrix(1, 1) = 5.0;
        eigen_matrix(1, 2) = 6.0;

        symmetry::Group g;
        sparsity::Pattern p;
        layout::MonoTile scalar_layout(shape::Smooth{}, g, p);
        layout::MonoTile vector_layout(shape::Smooth{2}, g, p);
        layout::MonoTile matrix_layout(shape::Smooth{2, 3}, g, p);

        buffer_type scalar(eigen_scalar, scalar_layout);
        buffer_type vector(eigen_vector, vector_layout);
        buffer_type matrix(eigen_matrix, matrix_layout);

        REQUIRE(compare_eigen(scalar.template value<0>(), eigen_scalar));
        REQUIRE(compare_eigen(vector.template value<1>(), eigen_vector));
        REQUIRE(compare_eigen(matrix.template value<2>(), eigen_matrix));
    }
}
