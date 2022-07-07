#include "../test_tensor.hpp"
#include <tensorwrapper/tensor/expression/detail_/labeled.hpp>
#include <tensorwrapper/tensor/expression/detail_/nnary.hpp>
#include <tensorwrapper/tensor/expression/labeled_view.hpp>

using namespace tensorwrapper::tensor;

/* Testing notes
 * - LabeledView classes ultimately trigger the evaluation of Expression
 *   objects. To test that this is done correctly, below we create a facade
 *   class, ExpressionTestPIMPL. This class includes checks to make sure that
 *   operator=(Expression) behaves correctly, namely that it passes the correct
 *   values to the Expression and that it properly consumes the values returned
 *   from the Expression. That the Expression object behaves correctly is
 *   tested elsewhere.
 * - A number of pieces of the LabeledView class simply make Expression objects.
 *   The unit tests in this file ensure that the Expression objects are set-up
 *   correctly. They do NOT test that the Expression objects work correctly.
 *   Testing that the Expression objects work correctly is done in the unit
 *   tests for the various classes deriving from ExpressionPIMPL (e.g.,
 *   the unit tests in detail_/add.cpp ensure that the addition expression
 *   works correctly)
 */

namespace testing {

// Facade used to test operator=(Expression)
template<typename FieldType>
struct ExpressionTestPIMPL
  : public expression::detail_::NNary<FieldType,
                                      ExpressionTestPIMPL<FieldType>> {
    using my_type   = ExpressionTestPIMPL<FieldType>;
    using base_type = expression::detail_::NNary<FieldType, my_type>;

    using labeled_tensor = expression::LabeledView<FieldType>;
    using typename base_type::const_allocator_reference;
    using typename base_type::const_label_reference;
    using typename base_type::const_shape_reference;
    using typename base_type::label_type;
    using typename base_type::tensor_type;

    /* This should be initialized with the tensor whose operator=(Expression)
     * method is being called.
     */
    ExpressionTestPIMPL(labeled_tensor* pt) : m_ptensor(pt) {}

    label_type labels_(const_label_reference labels) const override {
        return labels;
    }

    tensor_type tensor_(const_label_reference labels,
                        const_shape_reference shape,
                        const_allocator_reference alloc) const override {
        REQUIRE(labels == m_ptensor->labels());
        return m_ptensor->tensor();
    }

    labeled_tensor* m_ptensor;
};

} // namespace testing

TEMPLATE_LIST_TEST_CASE("LabeledView", "", testing::field_types) {
    using field_type      = TestType;
    using labeled_view    = expression::LabeledView<field_type>;
    using tensor_type     = typename labeled_view::tensor_type;
    using label_type      = typename labeled_view::label_type;
    using expression_type = typename labeled_view::expression_type;

    constexpr bool is_tot = std::is_same_v<TestType, field::Tensor>;

    auto tensors   = testing::get_tensors<field_type>();
    auto& v        = tensors.at(is_tot ? "vector-of-vectors" : "vector");
    const auto& cv = v;

    // Sanity check v and cv alias same instance
    REQUIRE(&v == &cv);

    // Make a label and some labeld_tensors
    label_type v_labels(is_tot ? "i;j" : "i");
    labeled_view lv(v_labels, v);
    labeled_view lcv(v_labels, cv);

    SECTION("CTors") {
        // For most of these we're just looking for the labels to have the right
        // value and the resulting instance to properly alias the tensor

        SECTION("Alias read/write") {
            REQUIRE(lv.labels() == v_labels);
            REQUIRE(&lv.tensor() == &v);
        }

        SECTION("Alias read-only") {
            REQUIRE(lcv.labels() == v_labels);
            REQUIRE(&std::as_const(lcv).tensor() == &v);
        }

        SECTION("Copy") {
            labeled_view lv_copy(lv);
            labeled_view lcv_copy(lcv);

            REQUIRE(lv_copy.labels() == v_labels);
            REQUIRE(&lv_copy.tensor() == &v);

            REQUIRE(lcv_copy.labels() == v_labels);
            REQUIRE(&std::as_const(lcv).tensor() == &v);
        }

        SECTION("Move") {
            labeled_view lv_moved(std::move(lv));
            labeled_view lcv_moved(std::move(lcv));

            REQUIRE(lv_moved.labels() == v_labels);
            REQUIRE(&lv_moved.tensor() == &v);

            REQUIRE(lcv_moved.labels() == v_labels);
            REQUIRE(&std::as_const(lv_moved).tensor() == &v);
        }
    }

    SECTION("expresion()") {
        // We assume here that the Labeled expression works correctly
        using labeled = expression::detail_::Labeled<field_type>;
        expression_type corr(std::make_unique<labeled>(lv));
        REQUIRE(lv.expression() == corr);

        expression_type const_corr(std::make_unique<labeled>(lcv));
        REQUIRE(lcv.expression() == const_corr);
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
        /* This operation just calls operator=(expression())
         *
         * We know that expression() works and the next section ensures
         * that operator=(Expression) works. Thus we mainly need to make sure
         * that the return is correct and that the call properly forwards data
         * into rv.
         */
        tensor_type rv;
        labeled_view lrv(v_labels, rv);

        auto prv = &(lrv = lv);

        // Returns *this
        REQUIRE(prv == &lrv);
        // Should still alias rv, and rv should be now set to v
        REQUIRE(&lrv.tensor() == &rv);
        REQUIRE(rv == v);

        // Can't write to a read-only tensor
        REQUIRE_THROWS_AS((lcv = lv), std::runtime_error);
    }

    SECTION("operator=(Expression)") {
        // See notes at top for details on how this gets tested
        using pimpl = testing::ExpressionTestPIMPL<field_type>;
        expression_type exp(std::make_unique<pimpl>(&lv));

        auto plv = &(lv = exp);

        // Returns *this
        REQUIRE(plv == &lv);

        // Still aliases v
        REQUIRE(&lv.tensor() == &v);

        // Throws if called w/ read-only tensor
        REQUIRE_THROWS_AS((lcv = exp), std::runtime_error);
    }

    SECTION("operator+(LabeledView)") {
        // that the expression actually works is tested in add.cpp
        auto corr_lv = lv.expression() + lv.expression();
        auto exp_lv  = lv + lv;
        REQUIRE(corr_lv == exp_lv);

        auto corr_lcv = lcv.expression() + lcv.expression();
        auto exp_lcv  = lcv + lcv;
        REQUIRE(corr_lcv == exp_lcv);
    }

    SECTION("operator-(LabeledView)") {
        // that the expression actually works is tested in subtract.cpp
        auto corr_lv = lv.expression() - lv.expression();
        auto exp_lv  = lv - lv;
        REQUIRE(corr_lv == exp_lv);

        auto corr_lcv = lcv.expression() - lcv.expression();
        auto exp_lcv  = lcv - lcv;
        REQUIRE(corr_lcv == exp_lcv);
    }

    SECTION("operator*(LabeledView") {
        // that the expression actualy works is tested in times.cpp
        auto corr_lv = lv.expression() * lv.expression();
        auto exp_lv  = lv * lv;
        REQUIRE(corr_lv == exp_lv);

        auto corr_lcv = lcv.expression() * lcv.expression();
        auto exp_lcv  = lcv * lcv;
        REQUIRE(corr_lcv == exp_lcv);
    }

    SECTION("operator*(double)") {
        // that the expression actually works is tested in scale.cpp
        auto corr_lv = lv.expression() * 3.14;
        auto exp_lv  = lv * 3.14;
        REQUIRE(corr_lv == exp_lv);

        auto corr_lcv = lcv.expression() * 3.14;
        auto exp_lcv  = lcv * 3.14;
        REQUIRE(corr_lcv == exp_lcv);
    }

    SECTION("operator*(double, LabeledView)") {
        auto corr_lv = lv.expression() * 3.14;
        auto exp_lv  = 3.14 * lv;
        REQUIRE(corr_lv == exp_lv);

        auto corr_lcv = lcv.expression() * 3.14;
        auto exp_lcv  = 3.14 * lcv;
        REQUIRE(corr_lcv == exp_lcv);
    }

    SECTION("operator==/operator!=") {
        // Same holding read/write
        labeled_view other_lv(v_labels, v);
        REQUIRE(lv == other_lv);
        REQUIRE_FALSE(lv != other_lv);

        // Same holding read-only
        labeled_view other_lcv(v_labels, cv);
        REQUIRE(lcv == other_lcv);
        REQUIRE_FALSE(lcv != other_lcv);

        // Different const-ness
        REQUIRE_FALSE(lv == lcv);
        REQUIRE(lv != lcv);

        // Different labels
        labeled_view diff_label(is_tot ? "j;i" : "j", v);
        REQUIRE_FALSE(lv == diff_label);
        REQUIRE(lv != diff_label);

        // Different tensors
        tensor_type other_v(v);
        labeled_view diff_tensor(v_labels, other_v);
        REQUIRE(other_v == v); // Sanity check ensuring just diff addresses
        REQUIRE_FALSE(lv == diff_tensor);
        REQUIRE(lv != diff_tensor);
    }
}
