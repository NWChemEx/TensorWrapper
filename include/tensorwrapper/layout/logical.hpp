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
    using base_type = LayoutBase;

public:
    /// Pull in base class's types
    using base_type::layout_pointer;
    using base_type::shape_pointer;
    using base_type::size_type;
    using base_type::sparsity_pointer;
    using base_type::symmetry_pointer;

    Logical(shape_pointer pshape, symmetry_pointer psymmetry,
            sparsity_pointer psparsity) :
      base_type(std::move(pshape), std::move(psymmetry), std::move(psparsity)) {
    }

protected:
    /// Implements clone by calling copy ctor
    layout_pointer clone_() const override {
        return std::make_unique<Logical>(*this);
    }

    /// Implements are_equal by calling are_equal_impl_
    bool are_equal_(const layout_base& rhs) const noexcept override {
        return are_equal_impl_<Logical>(rhs);
    }
};

} // namespace tensorwrapper::layout
