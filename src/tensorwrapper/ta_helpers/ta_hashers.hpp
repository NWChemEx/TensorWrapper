/*
 * Copyright 2022 NWChemEx-Project
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
#include "ta_headers.hpp"
#include "tensorwrapper/detail_/hashing.hpp"
#include <madness/world/safempi.h>

namespace TiledArray {
//------------------------------------------------------------------------------
// Hashing for TiledArray Classes
//------------------------------------------------------------------------------

/** @brief Enables hashing for TA Range class.
 *
 * Free function to enable hashing with BPHash library.
 *
 * @param[in] R Range object
 * @param[in, out] h tensorwrapper::detail_::Hasher object.
 */
inline void hash_object(const TA::Range& R, tensorwrapper::detail_::Hasher& h) {
    // To create a unique hash for this type with the default constructor
    const char* mytype = "TA::Range";
    h(mytype);
    // Alternative to above but no guarantee for reproducibility
    // h(typeid(tr).hash_code());
    for(std::size_t i = 0; i < R.rank(); ++i) {
        h(R.dim(i).first);
        h(R.dim(i).second);
    }
}

/** @brief Enables hashing for TA TiledRange1 class.
 *
 * Free function to enable hashing with BPHash library.
 *
 * @param[in] tr TiledRange1 object
 * @param[in, out] h tensorwrapper::detail_::Hasher object.
 */
inline void hash_object(const TA::TiledRange1& tr,
                        tensorwrapper::detail_::Hasher& h) {
    const char* mytype = "TA::TiledRange1";
    h(mytype);
    for(const auto& tile : tr) h(tile.first);
    h(tr.extent());
}

/** @brief Enables hashing for TA TiledRange class.
 *
 * Free function to enable hashing with BPHash library.
 *
 * @param[in] tr TiledRange object
 * @param[in, out] h tensorwrapper::detail_::Hasher object.
 */
inline void hash_object(const TiledArray::TiledRange& tr,
                        tensorwrapper::detail_::Hasher& h) {
    const char* mytype = "TA::TiledRange";
    h(mytype);
    for(std::size_t i = 0; i < tr.rank(); ++i) h(tr.dim(i));
}

/** @brief Enables hashing for TA Pmap class.
 *
 * Free function to enable hashing with BPHash library.
 *
 * @param[in] p Pmap object
 * @param[in, out] h tensorwrapper::detail_::Hasher object.
 */
inline void hash_object(const TiledArray::Pmap& p,
                        tensorwrapper::detail_::Hasher& h) {
    const char* mytype = "TA::Pmap";
    h(mytype);
    h(p.rank());
    h(p.size());
}

/** @brief Enables hashing for TA Tensor class.
 *
 * Free function to enable hashing with BPHash library.
 *
 * @tparam ValueType Type of value for the @p A tensor.
 * @tparam AllocatorType Type of allocator for the @p A tensor.
 * @param[in] A Tensor object
 * @param[in, out] h tensorwrapper::detail_::Hasher object.
 */
template<typename ValueType, typename AllocatorType>
void hash_object(const TA::Tensor<ValueType, AllocatorType>& A,
                 tensorwrapper::detail_::Hasher& h) {
    const char* mytype = "TA::Tensor";
    h(A.range());
    const auto n = A.range().volume();
    for(auto i = 0ul; i < n; ++i) h(A[i]);
}

/** @brief Returns a 128 bit hash value for TA::DistArray object.
 *
 * Involves MPI collective operation (MPI_Allreduce)
 *
 * @tparam TileType Type of tensor for @p A.
 * @tparam PolicyType Type of policy for @p A. Either DensePolicy or
 * SparsePolicy.
 * @param[in] A DistArray object
 * @return tensorwrapper::detail_::HashValue for TA::DistArray
 */
template<typename TensorType, typename PolicyType>
tensorwrapper::detail_::HashValue get_tile_hash_sum(
  const TA::DistArray<TensorType, PolicyType>& A) {
    auto& madworld = A.world();
    auto& mpiworld = madworld.mpi.Get_mpi_comm();

    // Note: Without the fence orbital space hash tests hang on parallel runs.
    madworld.gop.fence();
    tensorwrapper::detail_::HashValue myhash;
    tensorwrapper::detail_::HashValue mytotal(16, 0);
    tensorwrapper::detail_::HashValue hashsum(16, 0);
    // tensorwrapper::detail_::HashType::Hash128 has 16 uint8_t

    for(auto it = A.begin(); it != A.end(); ++it) {
        auto tile = A.find(it.index()).get();
        myhash    = tensorwrapper::detail_::make_hash(
          tensorwrapper::detail_::HashType::Hash128, tile);
        for(auto i = 0; i < myhash.size(); i++) { mytotal[i] += myhash[i]; }
    }
    if(madworld.size() > 1 && !A.pmap().get()->is_replicated()) {
        // Note: Using &mytotal, &hashsum gives Seg Fault 11.
        madworld.mpi.Allreduce(mytotal.data(), hashsum.data(), mytotal.size(),
                               MPI_UINT8_T, MPI_SUM);
    } else {
        hashsum = mytotal;
    }
    madworld.gop.fence();
    return hashsum;
}

/** @brief Enables hashing for TA DistArray class.
 *
 * Free function to enable hashing with BPHash library.
 *
 * @tparam TensorType Type of tensor (TA::DistArray) for @p A.
 * @tparam PolicyType Type of policy for @p A. Either DensePolicy or
 * SparsePolicy.
 * @param[in] A DistArray object
 * @param[in, out] h tensorwrapper::detail_::Hasher object.
 */

template<typename TensorType, typename PolicyType>
void hash_object(const TA::DistArray<TensorType, PolicyType>& A,
                 tensorwrapper::detail_::Hasher& h) {
    const char* mytype = "TA::DistArray";
    h(mytype);
    if(A.is_initialized()) {
        h(A.tiles_range());
        h(get_tile_hash_sum(A));
    }
}

/** @brief Enables comparison between TA tensor objects
 *
 * Free function to enable comparison between TA tensor objects.
 *
 * @tparam ValueType Type of TA::tensor for @p A.
 * @tparam AllocatorType Type of TA::tensor for @p A.
 * @tparam ValueType Type of TA::tensor for @p B.
 * @tparam AllocatorType Type of TA::tensor for @p B.
 * @param[in] A TA tensor object
 * @param[in] B TA tensor object
 */
template<typename ValueTypeA, typename AllocatorTypeA, typename ValueTypeB,
         typename AllocatorTypeB>
bool operator==(const TA::Tensor<ValueTypeA, AllocatorTypeA>& A,
                const TA::Tensor<ValueTypeB, AllocatorTypeB>& B) {
    return tensorwrapper::detail_::hash_objects(A) ==
           tensorwrapper::detail_::hash_objects(B);
}

/** @brief Enables comparison between TA tensor objects
 *
 * Free function to enable comparison between TA tensor objects.
 *
 * @tparam ValueType Type of TA::tensor for @p A.
 * @tparam AllocatorType Type of TA::tensor for @p A.
 * @tparam ValueType Type of TA::tensor for @p B.
 * @tparam AllocatorType Type of TA::tensor for @p B.
 * @param[in] A TA tensor object
 * @param[in] B TA tensor object
 */
template<typename ValueTypeA, typename AllocatorTypeA, typename ValueTypeB,
         typename AllocatorTypeB>
bool operator!=(const TA::Tensor<ValueTypeA, AllocatorTypeA>& A,
                const TA::Tensor<ValueTypeB, AllocatorTypeB>& B) {
    return !(A == B);
}
} // namespace TiledArray
