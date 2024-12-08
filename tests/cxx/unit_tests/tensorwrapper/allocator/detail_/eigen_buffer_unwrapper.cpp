#include "../../helpers.hpp"
#include "../../inputs.hpp"
#include <tensorwrapper/allocator/detail_/eigen_buffer_unwrapper.hpp>

using namespace tensorwrapper;

namespace {

template<typename FloatType, unsigned short Rank>
void check_tensor(Tensor& t) {
    using unwrapper         = allocator::detail_::EigenBufferUnwrapper;
    using variant_type      = typename unwrapper::variant_type;
    using eigen_buffer_type = typename unwrapper::buffer_type<FloatType, Rank>;

    auto& eigen_buffer = dynamic_cast<eigen_buffer_type&>(t.buffer());
    variant_type corr{eigen_buffer};
    REQUIRE(unwrapper::downcast(t.buffer()) == corr);
}

} // namespace

TEST_CASE("EigenBufferUnwrapper") {
    SECTION("Tensor<float, 0>") {
        Tensor scalar(testing::smooth_scalar_<float>());
        check_tensor<float, 0>(scalar);
    }

    SECTION("Tensor<float, 1>") {
        Tensor vector(testing::smooth_vector_<float>());
        check_tensor<float, 1>(vector);
    }

    SECTION("Tensor<float, 2>") {
        Tensor matrix(testing::smooth_matrix_<float>());
        check_tensor<float, 2>(matrix);
    }

    SECTION("Tensor<float, 3>") {
        Tensor t(testing::smooth_tensor3_<float>());
        check_tensor<float, 3>(t);
    }

    SECTION("Tensor<double, 0>") {
        Tensor scalar(testing::smooth_scalar());
        check_tensor<double, 0>(scalar);
    }

    SECTION("Tensor<double, 1>") {
        Tensor vector(testing::smooth_vector());
        check_tensor<double, 1>(vector);
    }

    SECTION("Tensor<double, 2>") {
        Tensor matrix(testing::smooth_matrix());
        check_tensor<double, 2>(matrix);
    }

    SECTION("Tensor<double, 3>") {
        Tensor t(testing::smooth_tensor3());
        check_tensor<double, 3>(t);
    }
}