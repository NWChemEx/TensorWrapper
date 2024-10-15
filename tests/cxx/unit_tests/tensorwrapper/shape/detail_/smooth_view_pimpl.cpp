#include "../../helpers.hpp"
#include <tensorwrapper/shape/detail_/smooth_alias.hpp>

/* Testing Strategy.
 *
 * At present the only thing actually implemented in SmoothViewPIMPL is
 * are_equal so that's all this test case tests.
 */

using namespace tensorwrapper::shape;

using types2test = std::pair<Smooth, const Smooth>;

TEMPLATE_LIST_TEST_CASE("SmoothViewPIMPL", "", types2test) {
    using pimpl_type = detail_::SmoothAlias<TestType>;
    using shape_type = std::decay_t<TestType>;
    shape_type scalar_shape{}, shape{1, 2, 3};

    pimpl_type scalar(scalar_shape);
    pimpl_type value(shape);

    SECTION("are_equal") {
        SECTION("Same") {
            REQUIRE(scalar.are_equal(pimpl_type(scalar_shape)));
            REQUIRE(value.are_equal(pimpl_type(shape)));
        }

        SECTION("Different rank") {
            shape_type rhs_shape{1};
            pimpl_type rhs(rhs_shape);
            REQUIRE_FALSE(scalar.are_equal(rhs));
        }

        SECTION("Different extents") {
            shape_type rhs_shape{2, 1, 3};
            pimpl_type rhs(rhs_shape);
            REQUIRE_FALSE(value.are_equal(rhs));
        }
    }
}