// #include "tensorwrapper/tensor/tensor.hpp"
// #include <catch2/catch.hpp>
// #include <tensorwrapper/tensor/detail_/ta_to_tw.hpp>

// /* Testing strategy:
//  *
//  * OpLayer simply exposes two functions downcast. Here we instantiate a
//  * LabeledTensorWrapper instance and make sure that downcast properly
//  downcasts
//  * to it.
//  */

// using namespace tensorwrapper::tensor;

// TEST_CASE("OpLayer") {
//     using tensor         = ScalarTensorWrapper;
//     using ta_tensor      = TA::TSpArrayD;
//     using labeled_tensor = detail_::LabeledTensorWrapper<tensor>;
//     using base_type      = detail_::OpLayer<labeled_tensor>;
//     using vector_il      = TA::detail::vector_il<int>;

//     auto& world = TA::get_default_world();
//     auto t      = detail_::ta_to_tw(ta_tensor(world, vector_il{1, 2, 3}));
//     labeled_tensor lt("i", t);

//     SECTION("downcast") {
//         base_type& as_base(lt);
//         labeled_tensor& as_derived = as_base.downcast();
//         REQUIRE(&lt == &as_derived);
//     }

//     SECTION("downcast const") {
//         const base_type& as_base(lt);
//         const labeled_tensor& as_derived = as_base.downcast();
//         REQUIRE(&lt == &as_derived);
//     }
// }
