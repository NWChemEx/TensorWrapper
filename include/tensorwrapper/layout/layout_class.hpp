#pragma once
#include <tensorwrapper/shape/shape_base.hpp>
#include <tensorwrapper/sparsity/pattern.hpp>
#include <tensorwrapper/symmetry/group.hpp>

namespace tensorwrapper::layout {

/** @brief Describes how the tensor is actually laid out.
 *
 */
class Layout {
public:
    using shape_type    = shape::ShapeBase;
    using symmetry_type = symmetry::Group;
    using sparsity_type = sparsity::Pattern;

private:
    using shape_pointer;

    shape_pointer m_shape_;
    symmetry_type m_symmetry_;
    sparsity_type m_sparsity_;
};

} // namespace tensorwrapper::layout
