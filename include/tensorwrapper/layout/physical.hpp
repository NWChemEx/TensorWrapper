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

/** @brief Specializes a LayoutBase for a layout describing how a tensor is
 *         actually laid out at runtime.
 *
 *  At present this class is largely a strong type, but eventually we expect it
 *  to hold details such as row major vs column major that matter for the
 *  physical layout, but not the logical layout.
 */
class Physical : public LayoutBase {
private:
    /// Type *this derives from
    using base_type = LayoutBase;

public:
    /// Pull in base class's types
    using base_type::layout_pointer;
    using base_type::size_type;

    /// Reuse base class's ctors
    using base_type::base_type;

protected:
    /// Implements clone by calling copy ctor
    layout_pointer clone_() const override {
        return std::make_unique<Physical>(*this);
    }

    /// Implements are_equal by calling are_equal_impl_
    bool are_equal_(const layout_base& rhs) const noexcept override {
        return are_equal_impl_<Physical>(rhs);
    }
};

} // namespace tensorwrapper::layout
