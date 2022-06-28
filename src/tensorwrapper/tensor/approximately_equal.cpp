#include "tensorwrapper/tensor/approx_equal.hpp"
#include "tensorwrapper/tensor/allclose.hpp"
#include "tensorwrapper/tensor/allocators/allocator.hpp"
#include "tensorwrapper/tensor/shapes/shapes.hpp"

namespace tensorwrapper::tensor {

    template<typename FieldType>	    
    bool are_approximately_equal(const TensorWrapper<FieldType>& actual, const TensorWrapper<FieldType>& ref, double rtol, double atol){
    return allclose(actual, ref, rtol, atol) && 
	   (actual.allocator() == ref.allocator()) &&
           (actual.shape() ==  ref.shape()) ;
    }

} // namespace tensorwrapper::tensor
