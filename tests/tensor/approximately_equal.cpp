#include "tensorwrapper/tensor/tensor.hpp"
#include "test_tensor.hpp"
#include <catch2/catch.hpp>

using namespace tensorwrapper::tensor;

TEST_CASE("Approximate Equality Comparison"){

    using field_type   = field::Scalar;
    using TWrapper     = TensorWrapper<field_type>;
    using shape_type   = typename TWrapper::shape_type;
    using extents_type = typename TWrapper::extents_type;
    auto vec_shape = std::make_unique<shape_type>(extents_type{3});

    using allocator::ta::Distribution;
    using allocator::ta::Storage;
    using allocator::ta::Tiling;
    
    /* relative tolerance */	
    auto rtol = 1.0E-10;
    /* actual tolerance */
    auto atol = 1.0E-8;	    

    /* 1-D tensors  dimension extent of 3 */
    ScalarTensorWrapper tensor_one{  -0.5157294715892564, 0.1709151888271797, 11.3448142827620728};
    ScalarTensorWrapper tensor_two{  -0.5157294715892563, 0.1709151888271787, 11.3448142827624728};
    ScalarTensorWrapper tensor_three{-0.5157294715892563, 0.1709151888271787,  0.0034567891011000};
    ScalarTensorWrapper tensor_four{ -0.5157294715892563, 0.1709151888271787,  0.0034587891011000};
    ScalarTensorWrapper tensor_five{ -0.5157294715892563, 0.1709151888271787, 11.3448342827620728};
    
    /* 1-D tensor, dimension extent of 4 */
    ScalarTensorWrapper tensor_six{  -0.5157294715892563, 0.1709151888271787, 11.3448242827620728, 12.3456789068889456};
    
    /* copy of tensor one elements but with different allocator attributes 
     * The code below has been adapted from ../tests/tensor/scalar_tensor_wrapper.cpp */
    auto other_alloc   = allocator::ta_allocator<field_type>(
      Storage::Core, Tiling::SingleElementTile, Distribution::Distributed);
    auto new_p = other_alloc->clone();
    const auto* pa = new_p.get();
    auto tensor_one_copy = tensor_one.pimpl().clone();
    tensor_one_copy->reallocate(new_p->clone());
    TWrapper tensor_seven(std::move(tensor_one_copy));

    SECTION("Allclose, allocator and shape comparisons all pass"){   	
        REQUIRE(are_approximately_equal(tensor_one, tensor_two, rtol, atol) == true);
    }
    SECTION("Allclose fails because of atol;  allocator and shape comparisons both pass"){   	
        REQUIRE(are_approximately_equal(tensor_one, tensor_five, rtol, atol) == false);
    }
    SECTION("Allclose fails because of rtol; allocator and shape comparisons both pass"){   	
        REQUIRE(are_approximately_equal(tensor_three, tensor_four, rtol, atol) == false);
    }
    SECTION("Allclose and allocator comparison pass, but shape comparison fails"){   	
        REQUIRE(are_approximately_equal(tensor_one, tensor_six, rtol, atol) == false);
    }
    SECTION("Allclose and shape comparison pass, but allocator comparison fails"){   	
        REQUIRE(are_approximately_equal(tensor_one, tensor_seven, rtol, atol) == false);
    }
}
