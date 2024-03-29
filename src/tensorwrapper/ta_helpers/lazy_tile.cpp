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

#include "lazy_tile.hpp"

namespace tensorwrapper::ta_helpers {

template class LazyTile<TA::Tensor<double>>;
template class LazyTile<TA::Tensor<TA::Tensor<double>>>;

/// Instantiate the static maps for the LazyTile types.
template<>
typename lazy_scalar_type::map_type lazy_scalar_type::evaluators{};

template<>
typename lazy_tot_type::map_type lazy_tot_type::evaluators{};

template<typename TileType>
LazyTile<TileType>::operator TileType() {
    return evaluators[m_id_](m_range_);
}

template<typename TileType>
void LazyTile<TileType>::add_evaluator(
  std::function<TileType(TA::Range)> evaluator, id_type id) {
    if(!evaluators.count(id)) evaluators[id] = evaluator;
}

/// Instantiate add_evaluator
template void lazy_scalar_type::add_evaluator(
  std::function<TA::Tensor<double>(TA::Range)> evaluator, id_type id);
template void lazy_tot_type::add_evaluator(
  std::function<TA::Tensor<TA::Tensor<double>>(TA::Range)> evaluator,
  id_type id);

/// Instantiate operator()
template lazy_scalar_type::operator TA::Tensor<double>();
template lazy_tot_type::operator TA::Tensor<TA::Tensor<double>>();

} // namespace tensorwrapper::ta_helpers