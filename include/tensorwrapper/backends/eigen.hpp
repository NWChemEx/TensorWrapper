#pragma once
#include "config.hpp"
#ifdef TENSORWRAPPER_HAS_EIGEN
#include <unsupported/Eigen/CXX11/Tensor>
#endif

namespace tensorwrapper::eigen {

#ifdef TENSORWRAPPER_HAS_EIGEN

template<typename FloatType, unsigned short Rank>
using tensor = Eigen::Tensor<FloatType, int(Rank)>;

#else

template<typename, unsigned short>
using tensor = std::nullptr_t;

#endif

} // namespace tensorwrapper::eigen
