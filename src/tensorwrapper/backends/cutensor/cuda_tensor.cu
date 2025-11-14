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
#include "cuda_tensor.cuh"
#include "cutensor_traits.cuh"
#include <unordered_map>
#include <vector>

namespace tensorwrapper::backends::cutensor {

// Some common typedefs
using mode_vector_t  = std::vector<int>;
using int64_vector_t = std::vector<int64_t>;

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
        }

// Convert a label into a vector of modes
template<typename LabelType>
mode_vector_t label_to_modes(const LabelType& label) {
    mode_vector_t mode;
    for(const auto& i : label) { mode.push_back(i.data()[0]); }
    return mode;
}

// Compute strides in row major
int64_vector_t get_strides(std::size_t N, const int64_vector_t& extent) {
    int64_vector_t strides;
    for(std::size_t i = 0; i < N; ++i) {
        int64_t product = 1;
        for(std::size_t j = N - 1; j > i; --j) product *= extent[j];
        strides.push_back(product);
    }
    return strides;
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

// Perform tensor contraction with cuTENSOR
template<typename TensorType>
void cutensor_contraction(typename TensorType::label_type c_label,
                          typename TensorType::label_type a_label,
                          typename TensorType::label_type b_label,
                          const TensorType& A, const TensorType& B,
                          TensorType& C) {
    using element_t = typename TensorType::value_type;

    const auto a_rank = A.rank();
    const auto b_rank = B.rank();
    const auto c_rank = C.rank();

    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();
    const auto& c_shape = C.shape();

    // GEMM alpha and beta (hardcoded for now)
    element_t alpha = 1.0;
    element_t beta  = 0.0;

    // The extents of each tensor
    int64_vector_t a_extents = get_extents(a_shape);
    int64_vector_t b_extents = get_extents(b_shape);
    int64_vector_t c_extents = get_extents(c_shape);

    // The modes of the tensors
    mode_vector_t a_modes = label_to_modes(a_label);
    mode_vector_t b_modes = label_to_modes(b_label);
    mode_vector_t c_modes = label_to_modes(c_label);

    // The strides of each tensor
    int64_vector_t a_strides = get_strides(a_rank, a_extents);
    int64_vector_t b_strides = get_strides(b_rank, b_extents);
    int64_vector_t c_strides = get_strides(c_rank, c_extents);

    // The size of each tensor
    std::size_t a_size = sizeof(element_t) * A.size();
    std::size_t b_size = sizeof(element_t) * B.size();
    std::size_t c_size = sizeof(element_t) * C.size();

    // Allocate on device
    void *A_d, *B_d, *C_d;
    cudaMalloc((void**)&A_d, a_size);
    cudaMalloc((void**)&B_d, b_size);
    cudaMalloc((void**)&C_d, c_size);

    // Copy to data to device
    HANDLE_CUDA_ERROR(
      cudaMemcpy(A_d, A.data(), a_size, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
      cudaMemcpy(B_d, B.data(), b_size, cudaMemcpyHostToDevice));
    HANDLE_CUDA_ERROR(
      cudaMemcpy(C_d, C.data(), c_size, cudaMemcpyHostToDevice));

    // Assert alignment
    const uint32_t kAlignment =
      128; // Alignment of the global-memory device pointers (bytes)
    assert(uintptr_t(A_d) % kAlignment == 0);
    assert(uintptr_t(B_d) % kAlignment == 0);
    assert(uintptr_t(C_d) % kAlignment == 0);

    // cuTENSOR traits
    cutensor_traits<element_t> traits;

    // cuTENSOR handle
    cutensorHandle_t handle;
    HANDLE_CUTENSOR_ERROR(cutensorCreate(&handle));

    // Create Tensor Descriptors
    cutensorTensorDescriptor_t descA;
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
      handle, &descA, a_rank, a_extents.data(), a_strides.data(),
      traits.cutensorDataType, kAlignment));

    cutensorTensorDescriptor_t descB;
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
      handle, &descB, b_rank, b_extents.data(), b_strides.data(),
      traits.cutensorDataType, kAlignment));

    cutensorTensorDescriptor_t descC;
    HANDLE_CUTENSOR_ERROR(cutensorCreateTensorDescriptor(
      handle, &descC, c_rank, c_extents.data(), c_strides.data(),
      traits.cutensorDataType, kAlignment));

    // Create Contraction Descriptor
    cutensorOperationDescriptor_t desc;
    HANDLE_CUTENSOR_ERROR(cutensorCreateContraction(
      handle, &desc,                               // Base
      descA, a_modes.data(), CUTENSOR_OP_IDENTITY, // A
      descB, b_modes.data(), CUTENSOR_OP_IDENTITY, // B
      descC, c_modes.data(), CUTENSOR_OP_IDENTITY, // C
      descC, c_modes.data(), traits.descCompute    // Result
      ));

    // Ensure that the scalar type is correct.
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

    // Determine workspace size and allocate
    uint64_t actualWorkspaceSize = 0;
    HANDLE_CUTENSOR_ERROR(cutensorPlanGetAttribute(
      handle, plan, CUTENSOR_PLAN_REQUIRED_WORKSPACE, &actualWorkspaceSize,
      sizeof(actualWorkspaceSize)));
    assert(actualWorkspaceSize <= workspaceSizeEstimate);

    void* work = nullptr;
    if(actualWorkspaceSize > 0) {
        HANDLE_CUDA_ERROR(cudaMalloc(&work, actualWorkspaceSize));
        assert(uintptr_t(work) % 128 ==
               0); // workspace must be aligned to 128 byte-boundary
    }

    // Execute
    cudaStream_t stream;
    HANDLE_CUDA_ERROR(cudaStreamCreate(&stream));
    HANDLE_CUTENSOR_ERROR(cutensorContract(handle, plan, (void*)&alpha, A_d,
                                           B_d, (void*)&beta, C_d, C_d, work,
                                           actualWorkspaceSize, stream));

    // Copy Results from Device
    HANDLE_CUDA_ERROR(
      cudaMemcpy(C.data(), C_d, c_size, cudaMemcpyDeviceToHost));

    // Free allocated memory
    HANDLE_CUTENSOR_ERROR(cutensorDestroy(handle));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyPlan(plan));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyOperationDescriptor(desc));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descA));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descB));
    HANDLE_CUTENSOR_ERROR(cutensorDestroyTensorDescriptor(descC));
    HANDLE_CUDA_ERROR(cudaStreamDestroy(stream));
    if(A_d) cudaFree(A_d);
    if(B_d) cudaFree(B_d);
    if(C_d) cudaFree(C_d);
    if(work) cudaFree(work);
}

#undef HANDLE_CUTENSOR_ERROR
#undef HANDLE_CUDA_ERROR

// Template instantiations
#define FUNCTION_INSTANTIATE(TYPE)                                    \
    template void cutensor_contraction<CUDATensor<TYPE>>(             \
      typename CUDATensor<TYPE>::label_type,                          \
      typename CUDATensor<TYPE>::label_type,                          \
      typename CUDATensor<TYPE>::label_type, const CUDATensor<TYPE>&, \
      const CUDATensor<TYPE>&, CUDATensor<TYPE>&)

TW_APPLY_FLOATING_POINT_TYPES(FUNCTION_INSTANTIATE);

#undef FUNCTION_INSTANTIATE

} // namespace tensorwrapper::backends::cutensor

#endif
