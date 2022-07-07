#include "tensorwrapper/tensor/tensor.hpp"
#include <catch2/catch.hpp>
#include <tensorwrapper/tensor/detail_/ta_to_tw.hpp>

using namespace tensorwrapper::tensor;

namespace {
TA::detail::vector_il<double> tensor_one_data{-0.5157294715892564, 0.1709151888271797,
                                        11.3448142827620728};
TA::detail::vector_il<double> tensor_two_data{-0.5157294715892563, 0.1709151888271787,
                                        11.3448142827624728};
}

TEST_CASE("Approximate Equality Comparison"){

    auto rtol = 1.0E-5;
    auto atol = 1.0E-8;	    
    using dvector_il = TA::detail::vector_il<double>;
    using TWrapper   = ScalarTensorWrapper;
    auto& world      = TA::get_default_world();
    
    const auto tensor_one = detail_::ta_to_tw(TA::TSpArrayD(world, tensor_one_data));
    const auto tensor_two = detail_::ta_to_tw(TA::TSpArrayD(world, tensor_two_data));
  
    SECTION("Approximate Equality"){   	
        REQUIRE(are_approximately_equal(tensor_one, tensor_two, rtol, atol));
    }

}
