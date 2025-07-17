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

namespace tensorwrapper::utilities {

/** @brief Produce a block diagonal matrix from a set of square matrices.
 *
 *  @param[in] matrices The vector of matrices to be placed along the output
 *                      diagonal.
 *
 *  @return A block diagonal matrix whose block values are equal to the input
 *          matrices.
 */
Tensor block_diagonal_matrix(std::vector<Tensor> matrices);

} // namespace tensorwrapper::utilities