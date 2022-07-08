// #include "../../test_tensor.hpp"
// #include <tensorwrapper/tensor/conversion/conversion.hpp>

// using namespace tensorwrapper;
// using namespace tensorwrapper::tensor;
// using scalar_tensor = field::Scalar;
// using tot_tensor    = field::Tensor;
// using ta_tot_tensor =
//   TA::DistArray<TA::Tensor<TA::Tensor<double>>, TA::SparsePolicy>;

// /* Testing Notes:
//  *
//  * For construction purposes we don't care if the labels make any sense. For
//  * example we're fine with `lhs("i0,i1") * rhs("i0;i1")` even though for a
//  * tensor times a ToT contraction can only occur over the independent
//  indices.
//  * What we are testing for construction purposes is that we've covered all of
//  * the possible products.
//  *
//  * Actual correctness of the contractions are spotted check later. It is
//  * assumed that the underlying tensor library can contract correctly so we
//  * don't have to check all of the possible contractions here too. What we are
//  * mainly interested in is that the TensorWrappers are const-correct and can
//  * be used to get the correct result.
//  */
// TEST_CASE("Tensor = Tensor * Tensor") {
//     using result_t       = TA::TSpArrayD;
//     using lhs_t          = scalar_tensor;
//     using rhs_t          = scalar_tensor;
//     using tensor_wrapper = ScalarTensorWrapper;

//     auto lhs = testing::get_tensors<lhs_t>().at("vector");
//     auto rhs = testing::get_tensors<rhs_t>().at("vector");

//     auto& world = TA::get_default_world();
//     result_t corr;

//     to_ta_distarrayd_t converter;
//     auto lhs_ta = converter.convert(lhs.pimpl().buffer());
//     auto rhs_ta = converter.convert(rhs.pimpl().buffer());
//     corr("i,j") = lhs_ta("i") * rhs_ta("j");

//     SECTION("all non-const") {
//         tensor_wrapper wrapped_lhs(lhs);
//         tensor_wrapper wrapped_rhs(rhs);
//         tensor_wrapper result;
//         result("i,j") = wrapped_lhs("i") * wrapped_rhs("j");
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }

//     SECTION("LHS is const") {
//         const tensor_wrapper wrapped_lhs(lhs);
//         tensor_wrapper wrapped_rhs(rhs);
//         tensor_wrapper result;
//         result("i,j") = wrapped_lhs("i") * wrapped_rhs("j");
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }

//     SECTION("RHS is const") {
//         tensor_wrapper wrapped_lhs(lhs);
//         const tensor_wrapper wrapped_rhs(rhs);
//         tensor_wrapper result;
//         result("i,j") = wrapped_lhs("i") * wrapped_rhs("j");
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }

//     SECTION("All are const") {
//         const tensor_wrapper wrapped_lhs(lhs);
//         const tensor_wrapper wrapped_rhs(rhs);
//         tensor_wrapper result;
//         result("i,j") = wrapped_lhs("i") * wrapped_rhs("j");
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }

//     SECTION("Mixed Hadamard and Contraction") {
//         auto lhs2 = testing::get_tensors<lhs_t>().at("matrix");
//         auto rhs2 = testing::get_tensors<rhs_t>().at("matrix");
//         const tensor_wrapper wrapped_lhs(lhs2);
//         const tensor_wrapper wrapped_rhs(rhs2);
//         tensor_wrapper result;
//         result("i") = wrapped_lhs("i,mu") * wrapped_rhs("mu,i");
//         TA::TSpArrayD corr(world, {7, 22});
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }
// }

// TEST_CASE("Tensor = ToT * ToT") {
//     using result_t       = TA::TSpArrayD;
//     using lhs_t          = tot_tensor;
//     using rhs_t          = tot_tensor;
//     using tensor_wrapper = ScalarTensorWrapper;
//     using tot_wrapper    = TensorOfTensorsWrapper;

//     auto lhs = testing::get_tensors<lhs_t>().at("vector-of-vectors");
//     auto rhs = testing::get_tensors<rhs_t>().at("vector-of-vectors");

//     auto& world = TA::get_default_world();
//     result_t corr;
//     to_ta_totd_t converter;
//     auto lhs_ta = converter.convert(lhs.pimpl().buffer());
//     auto rhs_ta = converter.convert(rhs.pimpl().buffer());
//     TA::expressions::einsum(corr("i"), lhs_ta("i;j"), rhs_ta("i;j"));

//     SECTION("all non-const") {
//         tot_wrapper wrapped_lhs(lhs);
//         tot_wrapper wrapped_rhs(rhs);
//         tensor_wrapper result;
//         result("i") = wrapped_lhs("i;j") * wrapped_rhs("i;j");
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }

//     SECTION("LHS is const") {
//         const tot_wrapper wrapped_lhs(lhs);
//         tot_wrapper wrapped_rhs(rhs);
//         tensor_wrapper result;
//         result("i") = wrapped_lhs("i;j") * wrapped_rhs("i;j");
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }

//     SECTION("RHS is const") {
//         tot_wrapper wrapped_lhs(lhs);
//         const tot_wrapper wrapped_rhs(rhs);
//         tensor_wrapper result;
//         result("i") = wrapped_lhs("i;j") * wrapped_rhs("i;j");
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }

//     SECTION("All are const") {
//         const tot_wrapper wrapped_lhs(lhs);
//         const tot_wrapper wrapped_rhs(rhs);
//         tensor_wrapper result;
//         result("i") = wrapped_lhs("i;j") * wrapped_rhs("i;j");
//         REQUIRE(ta_helpers::allclose(result.get<result_t>(), corr));
//     }
// }

// TEST_CASE("ToT = Tensor * ToT") {
//     using result_t         = ta_tot_tensor;
//     using lhs_t            = scalar_tensor;
//     using rhs_t            = tot_tensor;
//     using wrapped_result_t = TensorOfTensorsWrapper;
//     using wrapped_lhs_t    = ScalarTensorWrapper;
//     using wrapped_rhs_t    = TensorOfTensorsWrapper;

//     auto lhs = testing::get_tensors<lhs_t>().at("matrix");
//     auto rhs = testing::get_tensors<rhs_t>().at("matrix-of-vectors");
//     const auto result_idx = "i,j;k";
//     const auto lhs_idx    = "i,j";
//     const auto rhs_idx    = "i,j;k";

//     auto& world = TA::get_default_world();
//     result_t corr;
//     to_ta_distarrayd_t t_converter;
//     to_ta_totd_t tot_converter;
//     auto lhs_ta = t_converter.convert(lhs.pimpl().buffer());
//     auto rhs_ta = tot_converter.convert(rhs.pimpl().buffer());
//     TA::expressions::einsum(corr(result_idx), lhs_ta(lhs_idx),
//     rhs_ta(rhs_idx));

//     SECTION("all non-const") {
//         wrapped_lhs_t wrapped_lhs(lhs);
//         wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("LHS is const") {
//         const wrapped_lhs_t wrapped_lhs(lhs);
//         wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("RHS is const") {
//         wrapped_lhs_t wrapped_lhs(lhs);
//         const wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("all const") {
//         const wrapped_lhs_t wrapped_lhs(lhs);
//         const wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }
// }

// TEST_CASE("ToT = ToT * Tensor") {
//     using result_t         = ta_tot_tensor;
//     using lhs_t            = tot_tensor;
//     using rhs_t            = scalar_tensor;
//     using wrapped_result_t = TensorOfTensorsWrapper;
//     using wrapped_lhs_t    = TensorOfTensorsWrapper;
//     using wrapped_rhs_t    = ScalarTensorWrapper;

//     auto lhs = testing::get_tensors<lhs_t>().at("matrix-of-vectors");
//     auto rhs = testing::get_tensors<rhs_t>().at("matrix");
//     const auto result_idx = "i,j;k";
//     const auto lhs_idx    = "i,j;k";
//     const auto rhs_idx    = "i,j";

//     auto& world = TA::get_default_world();
//     result_t corr;
//     to_ta_distarrayd_t t_converter;
//     to_ta_totd_t tot_converter;
//     auto lhs_ta = tot_converter.convert(lhs.pimpl().buffer());
//     auto rhs_ta = t_converter.convert(rhs.pimpl().buffer());
//     TA::expressions::einsum(corr(result_idx), rhs_ta(rhs_idx),
//     lhs_ta(lhs_idx));

//     SECTION("all non-const") {
//         wrapped_lhs_t wrapped_lhs(lhs);
//         wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("LHS const") {
//         const wrapped_lhs_t wrapped_lhs(lhs);
//         wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("RHS const") {
//         wrapped_lhs_t wrapped_lhs(lhs);
//         const wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("all const") {
//         const wrapped_lhs_t wrapped_lhs(lhs);
//         const wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }
// }

// TEST_CASE("ToT = ToT * ToT") {
//     using result_t         = ta_tot_tensor;
//     using lhs_t            = tot_tensor;
//     using rhs_t            = tot_tensor;
//     using wrapped_result_t = TensorOfTensorsWrapper;
//     using wrapped_lhs_t    = TensorOfTensorsWrapper;
//     using wrapped_rhs_t    = TensorOfTensorsWrapper;

//     auto lhs = testing::get_tensors<lhs_t>().at("matrix-of-vectors");
//     auto rhs = testing::get_tensors<rhs_t>().at("matrix-of-vectors");
//     const auto result_idx = "i,j;k";
//     const auto lhs_idx    = "i,j;k";
//     const auto rhs_idx    = "i,j;k";

//     auto& world = TA::get_default_world();
//     result_t corr;
//     to_ta_totd_t tot_converter;
//     auto lhs_ta = tot_converter.convert(lhs.pimpl().buffer());
//     auto rhs_ta = tot_converter.convert(rhs.pimpl().buffer());
//     TA::expressions::einsum(corr(result_idx), lhs_ta(lhs_idx),
//     rhs_ta(rhs_idx));

//     SECTION("all non-const") {
//         wrapped_lhs_t wrapped_lhs(lhs);
//         wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("LHS const") {
//         const wrapped_lhs_t wrapped_lhs(lhs);
//         wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("RHS const") {
//         wrapped_lhs_t wrapped_lhs(lhs);
//         const wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }

//     SECTION("all const") {
//         const wrapped_lhs_t wrapped_lhs(lhs);
//         const wrapped_rhs_t wrapped_rhs(rhs);
//         wrapped_result_t result;
//         result(result_idx) = wrapped_lhs(lhs_idx) * wrapped_rhs(rhs_idx);
//         REQUIRE(ta_helpers::allclose_tot(result.get<result_t>(), corr, 1));
//     }
// }
