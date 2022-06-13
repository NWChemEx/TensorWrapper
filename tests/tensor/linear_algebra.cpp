#include "tensorwrapper/tensor/tensor.hpp"
#include <catch2/catch.hpp>
#include <tensorwrapper/tensor/detail_/ta_to_tw.hpp>

using namespace tensorwrapper::tensor;

namespace {
TA::detail::vector_il<double> eval_data{-0.5157294715892564, 0.1709151888271797,
                                        11.3448142827620728};

TA::detail::matrix_il<double> evec_data{
  {0.7369762290995787, 0.5910090485061027, 0.3279852776056817},
  {0.3279852776056812, -0.7369762290995785, 0.5910090485061033},
  {-0.5910090485061031, 0.3279852776056821, 0.7369762290995784}};

TA::detail::vector_il<double> svd_values{14.2274074126337418,
                                         1.2573298353791105};

TA::detail::matrix_il<double> svd_left{
  {-0.3761682344281408, -0.9265513797988839},
  {-0.9265513797988839, 0.3761682344281408}};

TA::detail::matrix_il<double> svd_right{
  {-0.3520616924890126, -0.4436257825895202, -0.5351898726900277,
   -0.6267539627905352},
  {0.7589812676751461, 0.3212415991459322, -0.1164980693832819,
   -0.5542377379124960}};

} // namespace

TEST_CASE("eigen_solve") {
    using ivector_il = TA::detail::vector_il<int>;
    using imatrix_il = TA::detail::matrix_il<int>;
    using dvector_il = TA::detail::vector_il<double>;
    using dmatrix_il = TA::detail::matrix_il<double>;
    using TWrapper   = ScalarTensorWrapper;
    auto& world      = TA::get_default_world();
    TA::TSpArrayD data(world,
                       imatrix_il{ivector_il{1, 2, 3}, ivector_il{2, 4, 5},
                                  ivector_il{3, 5, 6}});
    auto X = detail_::ta_to_tw(data);

    auto eval_corr = detail_::ta_to_tw(TA::TSpArrayD(world, eval_data));
    auto evec_corr = detail_::ta_to_tw(TA::TSpArrayD(world, evec_data));

    SECTION("No overlap matrix") {
        const auto& [evals, evecs] = eigen_solve(X);
        SECTION("eigen values") { REQUIRE(allclose(eval_corr, evals)); }
        SECTION("eigen vectors") { REQUIRE(abs_allclose(evec_corr, evecs)); }
    }
    SECTION("With overlap") {
        TA::TSpArrayD ovp(world, dmatrix_il{dvector_il{1.0, 0.0, 0.0},
                                            dvector_il{0.0, 1.0, 0.0},
                                            dvector_il{0.0, 0.0, 1.0}});
        auto S                     = detail_::ta_to_tw(ovp);
        const auto& [evals, evecs] = eigen_solve(X, S);
        SECTION("eigen values") { REQUIRE(allclose(eval_corr, evals)); }
        SECTION("eigen vectors") { REQUIRE(abs_allclose(evec_corr, evecs)); }
    }
}

TEST_CASE("SVD") {
    using ivector_il = TA::detail::vector_il<int>;
    using imatrix_il = TA::detail::matrix_il<int>;
    using TWrapper   = ScalarTensorWrapper;
    auto& world      = TA::get_default_world();

    auto values_corr = detail_::ta_to_tw(TA::TSpArrayD(world, svd_values));
    auto left_corr   = detail_::ta_to_tw(TA::TSpArrayD(world, svd_left));
    auto right_corr  = detail_::ta_to_tw(TA::TSpArrayD(world, svd_right));
    auto X           = detail_::ta_to_tw(TA::TSpArrayD(
      world, imatrix_il{ivector_il{1, 2, 3, 4}, ivector_il{5, 6, 7, 8}}));

    SECTION("Values") {
        const auto& S = SVDValues(X);
        REQUIRE(allclose(S, values_corr));
    }

    SECTION("Left") {
        const auto& [S, U] = SVDLeft(X);
        REQUIRE(allclose(S, values_corr));
        REQUIRE(abs_allclose(U, left_corr));
    }

    SECTION("Right") {
        const auto& [S, VT] = SVDRight(X);
        REQUIRE(allclose(S, values_corr));
        REQUIRE(abs_allclose(VT, right_corr));
    }

    SECTION("All") {
        const auto& [S, U, VT] = SVD(X);
        REQUIRE(allclose(S, values_corr));
        REQUIRE(abs_allclose(U, left_corr));
        REQUIRE(abs_allclose(VT, right_corr));
    }

    SECTION("Transpose Matrix") {
        // Make input tall instead of wide
        X("y,x") = X("x,y");

        // Make transposed vectors.
        TWrapper alt_left_corr, alt_right_corr;
        alt_right_corr("y,x") = left_corr("x,y");
        alt_left_corr("y,x")  = right_corr("x,y");

        const auto& [S, U, VT] = SVD(X);
        REQUIRE(allclose(S, values_corr));
        REQUIRE(abs_allclose(U, alt_left_corr));
        REQUIRE(abs_allclose(VT, alt_right_corr));
    }
}
