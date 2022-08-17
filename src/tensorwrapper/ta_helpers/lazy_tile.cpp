#include "lazy_tile.hpp"

namespace tensorwrapper::ta_helpers {

template class LazyTile<TA::Tensor<double>>;
template class LazyTile<TA::Tensor<TA::Tensor<double>>>;

/// Instantiate the static maps for the LazyTile types.
template<>
typename lazy_scalar_type::map_type lazy_scalar_type::evaluators{};

template<>
typename lazy_tot_type::map_type lazy_tot_type::evaluators{};

} // namespace tensorwrapper::ta_helpers