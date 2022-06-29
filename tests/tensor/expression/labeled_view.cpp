#include "../test_tensor.hpp"
#include <tensorwrapper/tensor/expression/detail_/labeled.hpp>
#include <tensorwrapper/tensor/expression/detail_/nnary.hpp>
#include <tensorwrapper/tensor/expression/labeled_view.hpp>

using namespace tensorwrapper::tensor;

namespace testing {

// Facade used to test operator=(Expression)
template<typename FieldType>
struct ExpressionTestPIMPL
  : public expression::detail_::NNary<FieldType,
                                      ExpressionTestPIMPL<FieldType>> {
    using my_type   = ExpressionTestPIMPL<FieldType>;
    using base_type = expression::detail_::NNary<FieldType, my_type>;

    using typename base_type::labeled_tensor;

    /* This should be initialized with the tensor whose operator=(Expression)
     * method is being called.
     */
    ExpressionTestPIMPL(labeled_tensor* pt) : m_ptensor(pt) {}

    labeled_tensor& eval_(labeled_tensor& lhs) const override {
        auto plhs = &lhs;
        REQUIRE(plhs == m_ptensor);
        return lhs;
    }

    labeled_tensor* m_ptensor;
};

} // namespace testing

TEST_CASE("LabeledView<field::Scalar>") {
    using field_type      = field::Scalar;
    using labeled_view    = expression::LabeledView<field_type>;
    using tensor_type     = typename labeled_view::tensor_type;
    using label_type      = typename labeled_view::label_type;
    using expression_type = typename labeled_view::expression_type;

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
             * that operator= works. Thus we mainly need to make sure that
             * the return is correct and that the call properly forwards rv.
             */
            tensor_type rv;
            labeled_view lrv(v_labels, rv);

            auto prv = &(lrv = lv);

            // Returns *this
            REQUIRE(prv == &lrv);
            // Still aliases rv
            REQUIRE(&lrv.tensor() == &rv);
            // rv is now set to v
            REQUIRE(rv == v);

            REQUIRE_THROWS_AS((lcv = lv), std::runtime_error);
        }

        SECTION("operator=(Expression)") {
            /* For this test we create a facade Expression to verify that
             * operator=(Expression) calls the expression correctly (the facade
             * is designed so that it knows what arguments should be coming in)
             * the remaining test is that it returns *this.
             */
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
            labeled_view diff_label("j", v);
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
}
