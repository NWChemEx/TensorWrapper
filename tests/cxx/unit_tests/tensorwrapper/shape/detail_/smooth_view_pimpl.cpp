#include "../../helpers.hpp"
#include <tensorwrapper/shape/detail_/smooth_view_pimpl.hpp>

using namespace tensorwrapper::shape;

using types2test = std::pair<Smooth, const Smooth>;

TEMPLATE_LIST_TEST_CASE("SmoothViewPIMPL", "", types2test) {
    using pimpl_type = detail_::SmoothViewPIMPL<TestType>;
    std::decay_t<TestType> defaulted_shape, shape{1, 2, 3};

    pimpl_type defaulted(defaulted_shape);
    pimpl_type value(shape);

    SECTION("CTor") {
        REQUIRE(defaulted.rank() == defaulted_shape.rank());
        REQUIRE(defaulted.size() == defaulted_shape.size());

        REQUIRE(value.rank() == shape.rank());
        REQUIRE(value.size() == shape.size());
    }
}