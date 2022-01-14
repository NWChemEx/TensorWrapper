#pragma once

#include "tensorwrapper/sparse_map/domain/domain.hpp"
#include "tensorwrapper/sparse_map/from_sparse_map.hpp"
#include "tensorwrapper/sparse_map/index.hpp"
#include "tensorwrapper/sparse_map/make_pair_map.hpp"
#include "tensorwrapper/sparse_map/sparse_map/sparse_map.hpp"

/** @brief Contains classes and functions related to SparseMap objects.
 *
 */
namespace tensorwrapper::sparse_map {

using SparseMapEE = SparseMap<ElementIndex, ElementIndex>;
using SparseMapET = SparseMap<ElementIndex, TileIndex>;
using SparseMapTE = SparseMap<TileIndex, ElementIndex>;
using SparseMapTT = SparseMap<TileIndex, TileIndex>;

} // namespace tensorwrapper::sparse_map
