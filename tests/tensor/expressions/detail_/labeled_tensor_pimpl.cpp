#include "../../test_tensor.hpp"
#include <tensorwrapper/tensor/expressions/detail_/labeled_tensor_pimpl.hpp>

using namespace tensorwrapper::tensor;

/* Testing Notes:
 *
 * - For the tensors we use the address of the buffer to check if things get
 *   moved and/or deep copied.
 */

TEST_CASE("LabeledTensorPIMPL<field::Scalar>") {
    // Get types we need for unit tests
    using field_type  = field::Scalar;
    using pimpl_type  = expressions::detail_::LabeledTensorPIMPL<field_type>;
    using label_type  = typename pimpl_type::label_type;
    using tensor_type = typename pimpl_type::tensor_type;

    // Make some default tensors
    auto tensors   = testing::get_tensors<field_type>();
    auto& v        = tensors.at("vector");
    const auto& cv = tensors.at("vector");

    // Sanity check v and cv alias same instance
    REQUIRE(&v == &cv);

    // Make a label
    label_type v_labels("i");

    // Construct a pimpl that is empty, aliasing the tensor, and copying tensor
    pimpl_type defaulted;
    pimpl_type lv(v_labels, v);
    pimpl_type lcv(v_labels, cv);

    SECTION("CTors") {
        SECTION("Default") {
            REQUIRE(defaulted.labels() == "");
            REQUIRE(defaulted.tensor() == tensor_type{});
        }

        SECTION("Value aliasing") {
            REQUIRE(lv.labels() == v_labels);
            REQUIRE(lv.tensor() == v);

            // Labels should be copied
            REQUIRE(&lv.labels() != &v_labels);

            // Tensor should be aliased
            REQUIRE(&lv.tensor() == &v);
        }

        SECTION("Value owning") {
            REQUIRE(lcv.labels() == v_labels);
            REQUIRE(lcv.tensor() == v);

            // Labels should be copied
            REQUIRE(&lcv.labels() != &v_labels);

            // Tensor should be copied
            REQUIRE(&lcv.tensor() != &cv);
        }

        SECTION("Move") {
            pimpl_type lv_moved(std::move(lv));

            /// Labels are correct
            REQUIRE(lv_moved.labels() == v_labels);

            // Still aliases original tensor
            REQUIRE(&lv_moved.tensor() == &v);

            // Address of the tensor's buffer (used to see if we moved tensor)
            auto ptensor_lcv = &lcv.tensor().buffer();

            pimpl_type lcv_moved(std::move(lcv));

            // Labels are correct
            REQUIRE(lcv_moved.labels() == v_labels);

            // Assume tensor was moved if buffer has same address
            REQUIRE(ptensor_lcv == &lcv_moved.tensor().buffer());
        }
    }

    SECTION("clone") {
        auto lv_clone = lv.clone();

        // Labels are equal and deep copies
        REQUIRE(lv_clone->labels() == lv.labels());
        REQUIRE(&lv_clone->labels()[0] != &lv.labels()[0]);

        // Tensors are equal and no longer alias of v
        REQUIRE(lv_clone->tensor() == v);
        REQUIRE(&lv_clone->tensor() != &v);
        REQUIRE(&lv_clone->tensor().buffer() != &v.buffer());

        auto lcv_clone = lcv.clone();

        // Labels are equal and deep copies
        REQUIRE(lcv_clone->labels() == lcv.labels());
        REQUIRE(&lcv_clone->labels()[0] != &lcv.labels()[0]);

        // Tensors are equal and deep copies
        REQUIRE(lcv_clone->tensor() == lcv.tensor());
        REQUIRE(&lcv_clone->tensor().buffer() != &lcv.tensor().buffer());
    }

    SECTION("labels") {
        REQUIRE(defaulted.labels() == "");
        REQUIRE(lv.labels() == v_labels);
        REQUIRE(lcv.labels() == v_labels);
    }

    SECTION("tensor()") {
        REQUIRE(defaulted.tensor() == tensor_type{});
        REQUIRE(&lv.tensor() == &v); // Should be alias
        REQUIRE(lcv.tensor() == cv);
    }

    SECTION("tensor() const") {
        REQUIRE(std::as_const(defaulted).tensor() == tensor_type{});
        REQUIRE(&std::as_const(lv).tensor() == &v); // Should be alias
        REQUIRE(std::as_const(lcv).tensor() == v);
    }
}
