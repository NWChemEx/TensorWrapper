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