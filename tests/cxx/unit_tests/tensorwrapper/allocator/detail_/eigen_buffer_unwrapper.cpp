#include "../../helpers.hpp"
#include "../../inputs.hpp"
#include <tensorwrapper/allocator/detail_/eigen_buffer_unwrapper.hpp>

using namespace tensorwrapper;

TEST_CASE("EigenBufferUnwrapper") {
    using unwrapper = allocator::detail_::EigenBufferUnwrapper;
    using ef0       = typename unwrapper::buffer_type<float, 0>;
    using ef1       = typename unwrapper::buffer_type<float, 1>;

    auto scalar = testing::smooth_scalar();
    auto vector = testing::smooth_vector();
}