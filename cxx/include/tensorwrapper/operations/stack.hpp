/*
 * Copyright 2026 NWChemEx-Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <tensorwrapper/tensor/tensor.hpp>
#include <vector>

namespace tensorwrapper::operations {

/** @brief Joins a series of same-shaped tensors along a new mode.
 *
 *  This function is modeled after `numpy.stack`. Given @f$N@f$ tensors, each
 *  with rank @f$r@f$ and shape @f$s@f$, the result is a rank @f$r+1@f$ tensor
 *  whose shape is @f$s@f$ with an extent of @f$N@f$ inserted at mode
 *  @p axis, i.e., `s[:axis] + [N] + s[axis:]` in Python notation. Fixing the
 *  index of the new mode to @f$k@f$ yields the @f$k@f$-th input tensor.
 *
 *  For example, stacking two vectors of length 3 with @p axis equal to 0
 *  yields a 2 by 3 matrix whose rows are the vectors, whereas @p axis equal
 *  to 1 yields a 3 by 2 matrix whose columns are the vectors.
 *
 *  Like `numpy.stack`, @p axis may be negative, in which case it counts from
 *  the end of the result's modes (e.g., -1 makes the new mode the last mode).
 *  Unlike `numpy.stack`, no type promotion occurs; all tensors must hold the
 *  same floating-point type.
 *
 *  @param[in] tensors The tensors to stack. Must be non-empty and every tensor
 *                     must have the same shape and floating-point type.
 *  @param[in] axis The mode of the result along which the tensors are
 *                  stacked. Must be in the range [-(r+1), r]. Defaults to 0.
 *
 *  @return A new tensor containing the stacked tensors.
 *
 *  @throw std::runtime_error if @p tensors is empty, if the tensors do not
 *                            all have the same shape, if @p axis is out of
 *                            range, if the tensors do not all hold the same
 *                            floating-point type, or if any tensor's buffer
 *                            is not contiguous. Strong throw guarantee.
 */
Tensor stack(const std::vector<Tensor>& tensors, int axis = 0);

} // namespace tensorwrapper::operations
