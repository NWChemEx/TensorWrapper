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
#include <cuda_runtime.h>
#include <cutensor.h>

namespace tensorwrapper::buffer::detail_ {

// Traits for cuTENSOR based on the floating point type
template<typename FloatType>
struct cutensor_traits {};

template<>
struct cutensor_traits<float> {
    cutensorDataType_t cutensorDataType     = CUTENSOR_R_32F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_32F;
};

template<>
struct cutensor_traits<double> {
    cutensorDataType_t cutensorDataType     = CUTENSOR_R_64F;
    cutensorComputeDescriptor_t descCompute = CUTENSOR_COMPUTE_DESC_64F;
};

} // namespace tensorwrapper::buffer::detail_

#endif