/*
 * Copyright 2024 NWChemEx Community
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

namespace tensorwrapper::sparsity {

/** @brief Base class for objects describing the sparsity of a tensor. */
class Pattern {
public:
    /** @brief Determines if *this and @p rhs describe the same sparsity
     *         pattern.
     *
     *  At present the sparsity component of TensorWrapper is a stub so this
     *  method always returns true.
     *
     *  @param[in] rhs The object to compare against.
     *
     *  @return True if *this is value equal to @p rhs and false otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator==(const Pattern& rhs) const noexcept { return true; }

    /** @brief Is *this different from @p rhs?
     *
     *  This class defines "different" as not value equal. See the description
     *  of operator== for the definition of value equal.
     *
     *  @param[in] rhs The object to compare against
     *
     *  @return False if *this and @p rhs are value equal and true otherwise.
     *
     *  @throw None No throw guarantee.
     */
    bool operator!=(const Pattern& rhs) const noexcept {
        return !((*this) == rhs);
    }
};

} // namespace tensorwrapper::sparsity
