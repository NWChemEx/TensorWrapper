/*
 * Copyright 2024 NWChemEx-Project
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
#include <tensorwrapper/layout/layout_base.hpp>

namespace tensorwrapper::layout {

/** @brief Specializes a LayoutBase for a layout describing how a user wants to
 *         view the tensor.
 *
 *  At present this class is largely a strong type.
 */
class Logical : public LayoutBase {
private:
    /// Type *this derives from
    using my_base_type = LayoutBase;

public:
    /// Pull in base class's types
    using my_base_type::const_shape_reference;
    using my_base_type::const_sparsity_reference;
    using my_base_type::const_symmetry_reference;
    using my_base_type::layout_pointer;
    using my_base_type::shape_pointer;
    using my_base_type::size_type;
    using my_base_type::sparsity_pointer;
    using my_base_type::symmetry_pointer;

    Logical(const_shape_reference shape, const_symmetry_reference symmetry,
            const_sparsity_reference sparsity) :
      my_base_type(shape, symmetry, sparsity) {}

    Logical(const_shape_reference shape) : my_base_type(shape) {}

    Logical(shape_pointer pshape, symmetry_pointer psymmetry,
            sparsity_pointer psparsity) :
      my_base_type(std::move(pshape), std::move(psymmetry),
                   std::move(psparsity)) {}

    Logical(shape_pointer pshape) : my_base_type(std::move(pshape)) {}

protected:
    /// Implements clone by calling copy ctor
    layout_pointer clone_() const override {
        return std::make_unique<Logical>(*this);
    }

    layout_base& assign_(const layout_base& rhs) override {
        return assign_impl_<Logical>(rhs);
    }

    /// Implements are_equal by calling are_equal_impl_
    bool are_equal_(const layout_base& rhs) const noexcept override {
        return are_equal_impl_<Logical>(rhs);
    }
};

} // namespace tensorwrapper::layout
