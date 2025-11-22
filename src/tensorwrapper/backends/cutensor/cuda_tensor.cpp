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

#include "cuda_tensor.hpp"

#ifdef ENABLE_CUTENSOR
#include "eigen_tensor.cuh"
#endif

namespace tensorwrapper::backends::cutensor {

#define TPARAMS template<typename FloatType>
#define CUDA_TENSOR CUDATensor<FloatType>

TPARAMS
void CUDA_TENSOR::contraction_assignment(label_type this_label,
                                         label_type lhs_label,
                                         label_type rhs_label,
                                         const_my_reference lhs,
                                         const_my_reference rhs) {
#ifdef ENABLE_CUTENSOR
    cutensor_contraction<my_type>(this_label, lhs_label, rhs_label, lhs, rhs,
                                  *this);
#else
    throw std::runtime_error(
      "cuTENSOR backend not enabled. Recompile with -DENABLE_CUTENSOR.");
#endif
}

#undef CUDA_TENSOR
#undef TPARAMS

template class CUDATensor<float>;
template class CUDATensor<double>;

} // namespace tensorwrapper::backends::cutensor
