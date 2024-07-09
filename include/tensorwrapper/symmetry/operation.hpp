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
#include <memory>
#include <tensorwrapper/detail_/polymorphic_base.hpp>

namespace tensorwrapper::symmetry {

/** @brief Common API for classes describing a symmetry operation.
 *
 *  The Group class interacts with the elements of the group through a common
 *  API. This class defines that API. The Operation class itself models a
 *  transformation which when applied to a tensor leaves the tensor unchanged.
 */
class Operation : public detail_::PolymorphicBase<Operation> {
public:
    /// Common base class for all symmetry operations
    using base_type = Operation;

    /// Type of a reference to an operation's base class
    using base_reference = base_type&;

    /// Type of a read-only reference to an operation's base class
    using const_base_reference = const base_type&;

    /// Type of a pointer to a symmetry Operation's base class
    using base_pointer = std::unique_ptr<base_type>;

    /// Type used to index tensor modes
    using mode_index_type = unsigned short;

    // -------------------------------------------------------------------------
    // -- Ctors, assignment, and dtor
    // -------------------------------------------------------------------------

    /// Defaulted no-throw dtor
    virtual ~Operation() noexcept = default;

    // -------------------------------------------------------------------------
    // - Properties
    // -------------------------------------------------------------------------

    bool is_identity() const noexcept { return is_identity_(); }

    // -------------------------------------------------------------------------
    // -- Utility methods
    // -------------------------------------------------------------------------

protected:
    /// Derived class should overwrite to implement is_identity
    virtual bool is_identity_() const noexcept = 0;
};

} // namespace tensorwrapper::symmetry
