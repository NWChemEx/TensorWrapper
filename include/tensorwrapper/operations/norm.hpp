#pragma once
#include <tensorwrapper/tensor/tensor.hpp>

namespace tensorwrapper::operations {

/** @brief Returns the infinity norm of @p t.
 *
 *  The infinity norm of the tensor @p t is the element of @p t with the
 *  largest absolute value.
 *
 *  @param[in] t The tensor to take the norm of.
 *
 *  @return The infinity norm of @p t.
 */
Tensor infinity_norm(const Tensor& t);

} // namespace tensorwrapper::operations