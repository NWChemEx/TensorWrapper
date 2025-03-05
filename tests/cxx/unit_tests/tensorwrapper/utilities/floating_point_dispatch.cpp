#include "../testing/testing.hpp"

using namespace tensorwrapper;
using namespace tensorwrapper::utilities;

struct Kernel {
    template<typename FloatType>
    void run(buffer::BufferBase& buffer) {
        auto corr = testing::eigen_matrix<FloatType>();
        REQUIRE(corr->are_equal(buffer));
    }

    template<typename FloatType>
    bool run(buffer::BufferBase& buffer, buffer::BufferBase& corr) {
        return corr.are_equal(buffer);
    }
};

TEMPLATE_LIST_TEST_CASE("floating_point_dispatch", "",
                        types::floating_point_types) {
    Kernel kernel;
    auto tensor = testing::eigen_matrix<TestType>();

    SECTION("Single input, no return") {
        floating_point_dispatch(kernel, *tensor);
    }

    SECTION("Two inputs and a return") {
        REQUIRE(floating_point_dispatch(kernel, *tensor, *tensor));
    }
}