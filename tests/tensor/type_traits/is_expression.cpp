// #include "tensorwrapper/tensor/tensor_wrapper.hpp"
// #include "tensorwrapper/tensor/type_traits/is_expression.hpp"
// #include <catch2/catch.hpp>

// using namespace tensorwrapper::tensor;

// TEST_CASE("is_expression_v") {
//     STATIC_REQUIRE_FALSE(is_expression_v<ScalarTensorWrapper>);
//     STATIC_REQUIRE(
//       is_expression_v<detail_::LabeledTensorWrapper<TensorOfTensorsWrapper>>);
// }

// TEST_CASE("enable_if_expression_t") {
//     using expr = detail_::LabeledTensorWrapper<TensorOfTensorsWrapper>;
//     using type = enable_if_expression_t<expr, int>;
//     STATIC_REQUIRE(std::is_same_v<type, int>);
// }
