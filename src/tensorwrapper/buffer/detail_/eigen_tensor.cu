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
#ifdef ENABLE_CUTENSOR
#include "cutensor_traits.cuh"
#include "eigen_tensor.cuh"
#include <unordered_map>
#include <vector>

namespace tensorwrapper::buffer::detail_ {

// Handle cuda errors
#define HANDLE_CUDA_ERROR(x)                                \
    {                                                       \
        const auto err = x;                                 \
        if(err != cudaSuccess) {                            \
            printf("Error: %s\n", cudaGetErrorString(err)); \
            exit(-1);                                       \
        }                                                   \
    };

// Handle cuTENSOR errors
#define HANDLE_CUTENSOR_ERROR(x)                                \
    {                                                           \
        const auto err = x;                                     \
        if(err != CUTENSOR_STATUS_SUCCESS) {                    \
            printf("Error: %s\n", cutensorGetErrorString(err)); \
            exit(-1);                                           \
        }                                                       \
    };

// Some common typedefs
using mode_vector_t  = std::vector<int>;
using int64_vector_t = std::vector<int64_t>;

// Convert a label into a vector of modes
template<typename LabelType>
mode_vector_t label_to_mode_vector(const LabelType& label) {
    mode_vector_t mode;
    for(const auto& i : label) { mode.push_back(i.data()[0]); }
    return mode;
}

// Query extent information from an input
template<typename InfoType>
int64_vector_t get_extents(const InfoType& info) {
    int64_vector_t extent;
    for(std::size_t i = 0; i < info.rank(); ++i) {
        extent.push_back((int64_t)info.extent(i));
    }
    return extent;
}

// Compute strides in row major
int64_vector_t row_major_strides(std::size_t N, const int64_vector_t& extent) {
    int64_vector_t strides;
    for(std::size_t i = 0; i < N; ++i) {
        int64_t product = 1;
        for(std::size_t j = N - 1; j > i; --j) product *= extent[j];
        strides.push_back(product);
    }
    return strides;
}

// Perform tensor contraction with cuTENSOR
template<typename TensorType>
void cutensor_contraction(
  typename TensorType::label_type olabel,
  typename TensorType::label_type llabel,
  typename TensorType::label_type rlabel,
  typename TensorType::const_shape_reference result_shape,
  typename TensorType::const_pimpl_reference lhs,
  typename TensorType::const_pimpl_reference rhs,
  typename TensorType::eigen_reference result) {
    using element_t    = typename TensorType::element_type;
    using eigen_data_t = typename TensorType::eigen_data_type;

    // GEMM alpha and beta (hardcoded for now)
    element_t alpha = 1.0;
    element_t beta  = 0.0;

    // The modes of the tensors
    mode_vector_t lhs_modes    = label_to_mode_vector(llabel);
    mode_vector_t rhs_modes    = label_to_mode_vector(rlabel);
    mode_vector_t output_modes = label_to_mode_vector(olabel);

    // The extents of each tensor
    int64_vector_t lhs_extents    = get_extents(lhs);
    int64_vector_t rhs_extents    = get_extents(rhs);
    int64_vector_t output_extents = get_extents(result_shape.as_smooth());

    // The strides of each tensor
    int64_vector_t lhs_strides = row_major_strides(lhs.rank(), lhs_extents);
    int64_vector_t rhs_strides = row_major_strides(rhs.rank(), rhs_extents);
    int64_vector_t output_strides =
      row_major_strides(result_shape.rank(), output_extents);

    // The size of each tensor
    std::size_t lhs_size    = sizeof(element_t) * lhs.size();
    std::size_t rhs_size    = sizeof(element_t) * rhs.size();
    std::size_t output_size = sizeof(element_t) * result_shape.size();

    // Allocate on device
    void *lhs_d, *rhs_d, *output_d;
    cudaMalloc((void**)&lhs_d, lhs_size);
    cudaMalloc((void**)&rhs_d, rhs_size);
    cudaMalloc((void**)&output_d, output_size);

    // Copy to data to device
    HANDLE_CUDA_ERROR(cudaMemcpy(lhs_d, lhs.get_immutable_data(), lhs_size,
                                 cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(cudaMemcpy(rhs_d, rhs.get_immutable_data(), rhs_size,
                                 cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
      cudaMemcpy(output_d, result.data(), output_size, cudaMemcpyHostToDevice));

    // Assert alignment
    const uint32_t kAlignment =
      128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(lhs_d) % kAlignment == 0);
    assert(uintptr_t(rhs_d) % kAlignment == 0);
    assert(uintptr_t(output_d) % kAlignment == 0);

    // cuTENSOR traits
    cutensor_traits<element_t> traits;

    // cuTENSOR handle
    cutensorHandle_t handle;
    HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));

    // Create Tensor Descriptors
    cutensorTensorDescriptor_t descLHS;
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
      handle, &descLHS, lhs.rank(), lhs_extents.data(), lhs_strides.data(),
      traits.cutensorDataType, kAlignment));

    cutensorTensorDescriptor_t descRHS;
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
      handle, &descRHS, rhs.rank(), rhs_extents.data(), rhs_strides.data(),
      traits.cutensorDataType, kAlignment));

    cutensorTensorDescriptor_t descOutput;
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
      handle, &descOutput, result_shape.rank(), output_extents.data(),
      output_strides.data(), traits.cutensorDataType, kAlignment));

    // Create Contraction Descriptor
    cutensorOperationDescriptor_t desc;
    HANDLE_CUTENSOR_ERROR(cutensorCreateContraction(
      handle, &desc,                                         // Base
      descLHS, lhs_modes.data(), CUTENSOR_OP_IDENTITY,       // A
      descRHS, rhs_modes.data(), CUTENSOR_OP_IDENTITY,       // B
      descOutput, output_modes.data(), CUTENSOR_OP_IDENTITY, // C
      descOutput, output_modes.data(), traits.descCompute    // Output
      ));

    // Optional (but recommended): ensure that the scalar type is correct.
    cutensorDataType_t scalarType;
    HANDLE_CUTENSOR_ERROR(cutensorOperationDescriptorGetAttribute(
      handle, desc, CUTENSOR_OPERATION_DESCRIPTOR_SCALAR_TYPE,
      (void*)&scalarType, sizeof(scalarType)));
    assert(scalarType == traits.cutensorDataType);

    // Set the algorithm to use
    const cutensorAlgo_t algo = CUTENSOR_ALGO_DEFAULT;
    cutensorPlanPreference_t planPref;
    HANDLE_CUTENSOR_ERROR(cutensorCreatePlanPreference(handle, &planPref, algo,
                                                       CUTENSOR_JIT_MODE_NONE));

    // Query workspace estimate
    uint64_t workspaceSizeEstimate = 0;
    const cutensorWorksizePreference_t workspacePref =
      CUTENSOR_WORKSPACE_DEFAULT;
    HANDLE_CUTENSOR_ERROR(cutensorEstimateWorkspaceSize(
      handle, desc, planPref, workspacePref, &workspaceSizeEstimate));

    // Create Contraction Plan
    cutensorPlan_t plan;
    HANDLE_CUTENSOR_ERROR(
      cutensorCreatePlan(handle, &plan, desc, planPref, workspaceSizeEstimate));

    // Optional: Query information about the created plan
    // query actually used workspace
    uint64_t actualWorkspaceSize = 0;
    HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(
      handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    // At this point the user knows exactly how much memory is need by the
    // operation and only the smaller actual workspace needs to be allocated
    void* work = nullptr;
    if(actualWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 ==
               0); // workspace must be aligned to 128 byte-boundary
    }

    // Execute
    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
    HANDLE_CUTENSOR_ERROR(
      cutensorContract(handle, plan, (void*)&alpha, lhs_d, rhs_d, (void*)&beta,
                       output_d, output_d, work, actualWorkspaceSize, stream));

    // Copy Results from Device
    HANDLE_CUDA_ERROR(
      cudaMemcpy(result.data(), output_d, output_size, cudaMemcpyDeviceToHost));

    // Free allocated memory
    HANDLE_CUTENSOR_ERROR(cutensorDestroy(handle));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descLHS));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descRHS));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descOutput));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
    if(lhs_d) cudaFree(lhs_d);
    if(rhs_d) cudaFree(rhs_d);
    if(output_d) cudaFree(output_d);
    if(work) cudaFree(work);
}

#undef HANDLE_CUTENSOR_ERROR
#undef HANDLE_CUDA_ERROR

// Template instantiations
#define FUNCTION_INSTANTIATE(TYPE, RANK)                         \
    template void cutensor_contraction<EigenTensor<TYPE, RANK>>( \
      typename EigenTensor<TYPE, RANK>::label_type,              \
      typename EigenTensor<TYPE, RANK>::label_type,              \
      typename EigenTensor<TYPE, RANK>::label_type,              \
      typename EigenTensor<TYPE, RANK>::const_shape_reference,   \
      typename EigenTensor<TYPE, RANK>::const_pimpl_reference,   \
      typename EigenTensor<TYPE, RANK>::const_pimpl_reference,   \
      typename EigenTensor<TYPE, RANK>::eigen_reference)

#define DEFINE_CUTENSOR_CONTRACTION(TYPE) \
    FUNCTION_INSTANTIATE(TYPE, 0);        \
    FUNCTION_INSTANTIATE(TYPE, 1);        \
    FUNCTION_INSTANTIATE(TYPE, 2);        \
    FUNCTION_INSTANTIATE(TYPE, 3);        \
    FUNCTION_INSTANTIATE(TYPE, 4);        \
    FUNCTION_INSTANTIATE(TYPE, 5);        \
    FUNCTION_INSTANTIATE(TYPE, 6);        \
    FUNCTION_INSTANTIATE(TYPE, 7);        \
    FUNCTION_INSTANTIATE(TYPE, 8);        \
    FUNCTION_INSTANTIATE(TYPE, 9);        \
    FUNCTION_INSTANTIATE(TYPE, 10)

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_CUTENSOR_CONTRACTION);

#undef DEFINE_CUTENSOR_CONTRACTION
#undef FUNCTION_INSTANTIATE

} // namespace tensorwrapper::buffer::detail_

#endif