#pragma once
#include "tensorwrapper/tensor/tensor_wrapper.hpp"
#include "tensorwrapper/tensor/approximately_equal.hpp"
#include "tensorwrapper/tensor/allclose.hpp"
#include "tensorwrapper/tensor/allocators/allocator.hpp"
#include "tensorwrapper/tensor/shapes/shapes.hpp"

namespace tensorwrapper::tensor {

/** @brief Compares two TensorWrapper instances for closeness.
 *
 *  This function will compare two tensors, @p actual and @p ref, elementwise and
 *  determine if all their values are close to one another, as specified 
 *  in the file "tensorwrapper/include/tensor/allclose.hpp". The function
 *  also compares shape and allocator attributes of the two tensors.
 *
 *  @param[in] ref The tensor you computed and are comparing against a
 *                    reference value.
 *  @param[in] ref The tensor which @p actual is being compared to. Should
 *                 be "the correct value".
 *  @param[in] rtol The maximum percent error (as a decimal) allowed for any
 *                  particular value. Assumed to be a positive decimal. Defaults
 *                  to 1.0E-5, *i.e.*, 0.001%.
 *  @param[in] atol The effective value of zero for comparisons. Assumed to be a
 *                  positive decimal less than 1.0. Defaults to 1.0E-8.
 *  @return True if @p actual is "close" to @p ref and false otherwise.
 */

 template<typename FieldType>	
 bool are_approximately_equal(const TensorWrapper<FieldType>& actual, const TensorWrapper<FieldType>& ref, double rtol, double atol){
    return allclose(actual, ref, rtol, atol) &&
           (actual.allocator() == ref.allocator()) &&
           (actual.shape() ==  ref.shape()) ;
    }

} // namespace tensorwrapper::tensor

