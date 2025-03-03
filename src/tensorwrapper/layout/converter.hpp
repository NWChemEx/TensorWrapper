#pragma once
#include <tensorwrapper/layout/logical.hpp>
#include <tensorwrapper/layout/physical.hpp>

namespace tensorwrapper::layout {

/** @brief Converts a logical layout into a physical layout. */
class Converter {
public:
    using logical_type            = Logical;
    using const_logical_reference = const logical_type&;
    using physical_type           = Physical;
    using physical_pointer        = std::unique_ptr<physical_type>;

    physical_pointer convert(const_logical_reference logical) {
        return std::make_unique<physical_type>(
          logical.shape(), logical.symmetry(), logical.sparsity());
    }
};

} // namespace tensorwrapper::layout