#include <catch2/catch.hpp>
#include <tensorwrapper/tensor/expression/detail_/add.hpp>
#include <tensorwrapper/tensor/expression/detail_/labeled.hpp>
#include <tensorwrapper/tensor/expression/detail_/scale.hpp>
#include <tensorwrapper/tensor/expression/detail_/subtract.hpp>
#include <tensorwrapper/tensor/expression/detail_/times.hpp>

using namespace tensorwrapper::tensor;
using namespace expression::detail_;

/* Testing Notes:
 *
 * - The NNary class implements the value ctor, clone, and are_equal methods for
 *   expressions derived from it.
 * - Implementations of labels, and tensor are unit tested in the respective
 *   derived class's unit test suite (e.g., Add::labels and Add::tensor are
 *   unit tested in add.cpp)
 * - Since NNary is designed to use CRTP, it's eaisest to test the class through
 *   a derived class instance. For these tests we create an instance of each
 *   derived class and test the methods through those classes.
 *
 */
TEST_CASE("NNary<field::Scalar>") {
    using field_type    = field::Scalar;
    using tensor_type   = TensorWrapper<field_type>;
    using add_type      = Add<field_type>;
    using labeled_type  = Labeled<field_type>;
    using scale_type    = Scale<field_type>;
    using subtract_type = Subtract<field_type>;
    using times_type    = Times<field_type>;

    tensor_type a{{1.0, 2.0}, {3.0, 4.0}};
    tensor_type b{{5.0, 6.0}, {7.0, 8.0}};

    auto la = a("i,j").expression();
    auto lb = b("i,j").expression();

    add_type add(la, lb);
    labeled_type labeled(a("i,j"));
    scale_type scale(la, 3.14);
    subtract_type subtract(la, lb);
    times_type times(la, lb);

    SECTION("clone") {
        auto add_clone = add.clone();
        REQUIRE(add.are_equal(*add_clone));

        auto labeled_clone = labeled.clone();
        REQUIRE(labeled.are_equal(*labeled_clone));

        auto scale_clone = scale.clone();
        REQUIRE(scale.are_equal(*scale_clone));

        auto subtract_clone = subtract.clone();
        REQUIRE(subtract.are_equal(*subtract_clone));

        auto times_clone = times.clone();
        REQUIRE(times.are_equal(*times_clone));
    }

    SECTION("arg") {
        REQUIRE(add.arg<0>() == la);
        REQUIRE(add.arg<1>() == lb);

        REQUIRE(labeled.arg<0>() == a("i,j"));

        REQUIRE(scale.arg<0>() == la);
        REQUIRE(scale.arg<1>() == 3.14);

        REQUIRE(subtract.arg<0>() == la);
        REQUIRE(subtract.arg<1>() == lb);

        REQUIRE(times.arg<0>() == la);
        REQUIRE(times.arg<1>() == lb);
    }

    SECTION("are_equal") {
        // Note we need to check the are_equal detects differences at each
        // argument position (i.e., changing either of the arguments to a binary
        // expression is caught) and that are_equal can detect different derived
        // types

        REQUIRE(add.are_equal(add_type(la, lb)));
        REQUIRE_FALSE(add.are_equal(add_type(lb, lb)));
        REQUIRE_FALSE(add.are_equal(add_type(la, la)));
        REQUIRE_FALSE(add.are_equal(subtract));

        REQUIRE(labeled.are_equal(labeled_type(a("i,j"))));
        REQUIRE_FALSE(labeled.are_equal(labeled_type(a("j,i"))));
        REQUIRE_FALSE(labeled.are_equal(add));

        REQUIRE(scale.are_equal(scale_type(la, 3.14)));
        REQUIRE_FALSE(scale.are_equal(scale_type(lb, 3.14)));
        REQUIRE_FALSE(scale.are_equal(scale_type(la, 1.23)));
        REQUIRE_FALSE(scale.are_equal(add));

        REQUIRE(subtract.are_equal(subtract_type(la, lb)));
        REQUIRE_FALSE(subtract.are_equal(subtract_type(la, la)));
        REQUIRE_FALSE(subtract.are_equal(subtract_type(lb, lb)));
        REQUIRE_FALSE(subtract.are_equal(add));

        REQUIRE(times.are_equal(times_type(la, lb)));
        REQUIRE_FALSE(times.are_equal(times_type(la, la)));
        REQUIRE_FALSE(times.are_equal(times_type(lb, lb)));
        REQUIRE_FALSE(times.are_equal(add));
    }
}
