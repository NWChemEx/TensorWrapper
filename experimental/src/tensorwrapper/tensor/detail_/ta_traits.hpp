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
#include "../../ta_helpers/lazy_tile.hpp"
#include "tensorwrapper/tensor/fields.hpp"
#include <TiledArray/dist_array.h>
#include <TiledArray/tensor.h>
#include <variant>

/** @file ta_traits.hpp
 *
 *  @brief Contains traits for when TiledArray is used as the backend.
 *
 *  These traits classes basically exist to attempt to isolate the TensorWrapper
 *  library from the guts of TA (to an extent). What really matters in all of
 *  these traits types is the `variant_type` member as that's what is used by
 *  the TensorWrapper library.
 */

namespace tensorwrapper::tensor::detail_ {

/** @brief Primary template for establishing the types associated with
 *         TiledArray.
 *
 *  The primary template is not defined. Instead TiledArrayTraits is specialized
 *  for each field, establishing a mapping from a field to the types Tiled Array
 *  uses for that field.
 *
 *  @tparam T The type of the field the tensor is over.
 *
 */
template<typename T>
struct TiledArrayTraits;

/** @brief Specializes TiledArrayTraits for tensors which have scalar elements.
 *
 *  This specialization is selected when the tensor uses `field::Scalar`. The
 *  elements of such a tensor are floating point values.
 */
template<>
struct TiledArrayTraits<field::Scalar> {
    /// Typedef of the tile for a tensor of scalars
    template<typename T>
    using tensor_tile_type = TA::Tensor<T>;

    /// Typedef of a lazy tile of scalars
    template<typename T>
    using lazy_tile_type =
      tensorwrapper::ta_helpers::LazyTile<tensor_tile_type<T>>;

    /// Typedef of the tensor class
    template<typename T>
    using tensor_type = TA::DistArray<tensor_tile_type<T>, TA::SparsePolicy>;

    /// Typedef of a lazy tensor class
    template<typename T>
    using lazy_tensor_type = TA::DistArray<lazy_tile_type<T>, TA::SparsePolicy>;

    /// Type of a variant with all possible non-hierarchal tensor types in it
    using variant_type =
      std::variant<tensor_type<double>, lazy_tensor_type<double>>;
};

/** @brief Specializes TiledArrayTraits for tensors which have tensor elements.
 *
 *  This specialization is selected when the tensor uses `field::Tensor`. The
 *  elements of such a tensor are other tensors.
 */
template<>
struct TiledArrayTraits<field::Tensor> {
    /// Typedef of the tiles in a tensor-of-tensors
    template<typename T>
    using tensor_tile_type = TA::Tensor<TA::Tensor<T>>;

    /// Typedef of a lazy tile of scalars
    template<typename T>
    using lazy_tile_type =
      tensorwrapper::ta_helpers::LazyTile<tensor_tile_type<T>>;

    /// Typedef of the tensor-of-tensors class
    template<typename T>
    using tensor_type = TA::DistArray<tensor_tile_type<T>, TA::SparsePolicy>;

    /// Typedef of a lazy tensor class
    template<typename T>
    using lazy_tensor_type = TA::DistArray<lazy_tile_type<T>, TA::SparsePolicy>;

    /// Type of a variant with all possible hierarchal tensor_types in it
    using variant_type =
      std::variant<tensor_type<double>, lazy_tensor_type<double>>;
};

} // namespace tensorwrapper::tensor::detail_