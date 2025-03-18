#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/operations/norm.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::operations;

TEMPLATE_LIST_TEST_CASE("infinity_norm", "", types::floating_point_types) {
    SECTION("scalar") {
        shape::Smooth s{};
        Tensor scalar(s, testing::eigen_scalar<TestType>());
        auto norm = infinity_norm(scalar);
        REQUIRE(approximately_equal(scalar, norm));
    }

    SECTION("vector") {
        shape::Smooth s{5};
        Tensor vector(s, testing::eigen_vector<TestType>());
        Tensor corr(shape::Smooth{}, testing::eigen_scalar<TestType>(4));
        auto norm = infinity_norm(vector);
        REQUIRE(approximately_equal(corr, norm));
    }

    SECTION("matrix") {
        shape::Smooth s{2, 2};
        Tensor matrix(s, testing::eigen_matrix<TestType>());
        Tensor corr(shape::Smooth{}, testing::eigen_scalar<TestType>(4));
        auto norm = infinity_norm(matrix);
        REQUIRE(approximately_equal(corr, norm));
    }

    SECTION("rank 3 tensor") {
        shape::Smooth s{2, 2, 2};
        Tensor t(s, testing::eigen_tensor3<TestType>());
        Tensor corr(shape::Smooth{}, testing::eigen_scalar<TestType>(8));
        auto norm = infinity_norm(t);
        REQUIRE(approximately_equal(corr, norm));
    }

    SECTION("rank 4 tensor") {
        shape::Smooth s{2, 2, 2, 2};
        Tensor t(s, testing::eigen_tensor4<TestType>());
        Tensor corr(shape::Smooth{}, testing::eigen_scalar<TestType>(16));
        auto norm = infinity_norm(t);
        REQUIRE(approximately_equal(corr, norm));
    }
}