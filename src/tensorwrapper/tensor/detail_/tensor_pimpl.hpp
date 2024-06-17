#pragma once
#include <tensorwrapper/tensor/tensor_class.hpp>

namespace tensorwrapper::detail_ {

class TensorPIMPL {
public:
    /// Type *this implements
    using parent_type = Tensor;

    /// Type of a pointer to *this
    using pimpl_pointer = typename parent_type::pimpl_pointer;
};

} // namespace tensorwrapper::detail_
