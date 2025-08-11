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
#include <boost/container_hash/hash.hpp>
#include <tensorwrapper/types/floating_point.hpp>

/** @namespace tensorwrapper::buffer::detail_::hash_utilities
 *  @brief Utilities for hashing EigenTensor instances
 */
namespace tensorwrapper::buffer::detail_::hash_utilities {

/// The type of a hash
using hash_type = std::size_t;

/** @brief Adds the hash of an input value to the seed hash
 *
 *  @tparam InputType The type of the value being added to the hash
 *  @param[in,out] seed The initial value of the hash, which is overwritten when
 *                      the new value is added.
 *  @param[in] value The new value being hashed and combined with the @p seed.
 *
 *  @return The updated hash value
 *
 *  @throw none No throw guarantee
 */
template<typename InputType>
void hash_input(hash_type& seed, const InputType& value) {
    boost::hash_combine(seed, value);
}

#ifdef ENABLE_SIGMA

/** @brief Specialization for sigma::Uncertain values
 *
 *  @tparam T The floating point type of the uncertain value
 *  @param[in,out] seed The initial value of the hash, which is overwritten when
 *                      the new value is added.
 *  @param[in] value The new uncertain value being hashed and combined with the
 *                   seed.
 *
 *  @return The updated hash value
 *
 *  @throw none No throw guarantee
 */
template<typename T>
void hash_input(hash_type& seed, const sigma::Uncertain<T>& value) {
    hash_input(seed, value.mean());
    hash_input(seed, value.sd());
    for(const auto& [dep, deriv] : value.deps()) {
        hash_input(seed, dep);
        hash_input(seed, deriv);
    }
}

#endif

} // namespace tensorwrapper::buffer::detail_::hash_utilities
