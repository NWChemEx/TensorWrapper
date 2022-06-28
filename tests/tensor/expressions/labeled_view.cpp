#include "../test_tensor.hpp"
#include <tensorwrapper/tensor/expressions/labeled_view.hpp>

using namespace tensorwrapper::tensor;

TEST_CASE("LabeledView<field::Scalar>") {
    using field_type   = field::Scalar;
    using labeled_view = expressions::LabeledView<field_type>;
    using tensor_type  = typename labeled_view::tensor_type;
    using label_type   = typename labeled_view::label_type;

    auto tensors   = testing::get_tensors<field_type>();
    auto& v        = tensors.at("vector");
    const auto& cv = tensors.at("vector");

    // Sanity check v and cv alias same instance
    REQUIRE(&v == &cv);

    // Make a label and some labeld_tensors
    label_type v_labels("i");
    labeled_view lv(v_labels, v);
    labeled_view lcv(v_labels, cv);

    SECTION("CTors") {
        SECTION("Alias read/write") {
            REQUIRE(lv.labels() == v_labels);

            // Should be an alias
            REQUIRE(&lv.tensor() == &v);
        }

        SECTION("Alias read-only") {
            REQUIRE(lcv.labels() == v_labels);

            // Should be an alias
            REQUIRE(&std::as_const(lcv).tensor() == &v);
        }

        SECTION("Copy") {
            labeled_view lv_copy(lv);
            labeled_view lcv_copy(lcv);

            REQUIRE(lv_copy.labels() == v_labels);
            // Still aliases the same tensor
            REQUIRE(&lv_copy.tensor() == &v);

            REQUIRE(lcv_copy.labels() == v_labels);
            // Still aliases the same tensor
            REQUIRE(&std::as_const(lcv).tensor() == &v);
        }

        SECTION("Move") {
            labeled_view lv_moved(std::move(lv));
            labeled_view lcv_moved(std::move(lcv));

            REQUIRE(lv_moved.labels() == v_labels);
            // Still aliases
            REQUIRE(&lv_moved.tensor() == &v);

            REQUIRE(lcv_moved.labels() == v_labels);
            // Still aliases
            REQUIRE(&std::as_const(lv_moved).tensor() == &v);
        }

        SECTION("expresion()") {
            // To test that the expression returned by expression() is correct
            // we evaluate it and check that rv is populated correctly. This
            // assumes that the returned expression is a Labeled instance and
            // that the Labeled class works correctly
            tensor_type rv;
            labeled_view lrv(v_labels, rv);

            lv.expression().eval(lrv);
            REQUIRE(lrv.labels() == v_labels);
            // Still aliases
            REQUIRE(&lrv.tensor() == &rv);
            // Set to correct value
            // REQUIRE(rv.tensor() == v);
        }

        SECTION("tensor()") {
            REQUIRE(&lv.tensor() == &v);
            REQUIRE_THROWS_AS(lcv.tensor(), std::runtime_error);
        }

        SECTION("tensor() const") {
            REQUIRE(&std::as_const(lv).tensor() == &v);
            REQUIRE(&std::as_const(lcv).tensor() == &v);
        }

        SECTION("labels()") {
            REQUIRE(lv.labels() == v_labels);
            REQUIRE(lcv.labels() == v_labels);
        }

        SECTION("operator=(LabeledView)") {
            SECTION("LHS is empty tensor") {
                tensor_type rv;
                labeled_view lrv(v_labels, rv);

                SECTION("RHS is read/write") {
                    auto prv = &(lrv = lv);

                    // Returns *this
                    REQUIRE(prv == &lrv);
                    // Still aliases v
                    REQUIRE(&lrv.tensor() == &rv);
                    // rv is now set to v
                    REQUIRE(rv == v);
                }

                SECTION("empty tensor = read-only") {
                    auto prv = &(lrv = lcv);

                    // Returns *this
                    REQUIRE(prv == &lrv);
                    // Still aliases v
                    REQUIRE(&lrv.tensor() == &rv);
                    // rv is now set to v
                    REQUIRE(rv == v);
                }
            }

            SECTION("LHS is non-empty tensor") {
                tensor_type rv({101.0, 102.0, 103.0});
                labeled_view lrv(v_labels, rv);
                auto prv = &(lrv = lv);

                REQUIRE(rv == v);
            }
        }
    }
}
