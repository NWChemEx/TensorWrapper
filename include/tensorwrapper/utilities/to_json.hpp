#pragma once
#include <ostream>
#include <tensorwrapper/tensor/tensor.hpp>
namespace tensorwrapper::utilities {

/** @brief Adds a JSON representation of @p to @p os.
 *
 *  This function can be used to print a tensor out in a JSON format. For dense
 *  tensors this will be as a list of lists such that number of nestings is
 *  equal to the rank of the tensor.
 *
 *  @param[in,out] os The stream to print @p t to. After the function is called
 *                    @p os will contain the JSON representation of @p t.
 *  @param[in] t The tensor to print to @p os.
 *
 *  @note Since the caller controls the stream that is passed in, it is assumed
 *        that the caller has set the stream up to print floating point values
 *        in their desired format. For example,  do
 *        `os << std::fixed << std::setprecision(8);` prior to calling `to_json`
 *        to guarantee all floating point values are printed with 8 decimal
 *        places.
 *
 *  @return This function returns @p os by reference to support chaining.
 */
std::ostream& to_json(std::ostream& os, const Tensor& t);

} // namespace tensorwrapper::utilities