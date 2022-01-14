#include "shape_pimpl.hpp"

namespace tensorwrapper::tensor::detail_ {

template class ShapePIMPL<field::Scalar>;
template class ShapePIMPL<field::Tensor>;

} // namespace tensorwrapper::tensor::detail_
