/*
 * Copyright 2025 NWChemEx-Project
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
#ifdef ENABLE_CUTENSOR
#include "cuda_tensor.hpp"

namespace tensorwrapper::backends::cutensor {

/** @brief Performs a tensor contraction on GPU
 *
 *  @param[in] olabel The labels for the modes of the output.
 *  @param[in] llabel The labels for the modes of the left hand tensor.
 *  @param[in] rlabel The labels for the modes of the right hand tensor.
 *  @param[in] result_shape The intended shape of the result.
 *  @param[in] lhs The left hand tensor.
 *  @param[in] rhs The right hand tensor.
 *  @param[in, out] result The eigen tensor where the results are stored.
 *
 *  @throw std::bad_alloc if there is a problem allocating the copy of
 *                        @p layout. Strong throw guarantee.
 */
template<typename TensorType>
void cutensor_contraction(typename TensorType::label_type c_label,
                          typename TensorType::label_type a_label,
                          typename TensorType::label_type b_label,
                          const TensorType& A, const TensorType& B,
                          TensorType& C);

} // namespace tensorwrapper::backends::cutensor

#endif
