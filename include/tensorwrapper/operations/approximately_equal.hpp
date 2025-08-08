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
#include <tensorwrapper/tensor/tensor.hpp>

namespace tensorwrapper::operations {

/** @brief Determines if @p lhs equals @p rhs to within @p tol.
 *
 *  Two tensors are approximately equal if:
 *  - They are the same rank.
 *  - If for each element in @p lhs the corresponding element in @p rhs d
 *
 *
 *  @param[in] lhs The first tensor being compared.
 *  @param[in] rhs The second tensor being compared.
 *  @param[in] tol The absolute tolerance for establishing equality. Two
 *                 elements that differ by less than @p tol are equal.
 *
 *  @return True if @p lhs is approximately equal to @p rhs and false otherwise.
 *
 *  @throw std::runtime_error if the tensors do not contain doubles. Strong
 *                            throw guarantee.
 *
 */
bool approximately_equal(const Tensor& lhs, const Tensor& rhs,
                         double tol = 1e-16);

} // namespace tensorwrapper::operations
