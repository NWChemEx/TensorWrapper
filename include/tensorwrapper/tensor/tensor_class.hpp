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

namespace tensorwrapper {
namespace detail_ {
class TensorPIMPL;
}

class Tensor {
public:
    /// Type of the object implementing *this
    using pimpl_type = detail_::TensorPIMPL;

    /// Type of a pointer to an object of type pimpl_type
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    Tensor() noexcept;

    /// Defaulted no-throw dtor
    ~Tensor() noexcept;

private:
    /// Does *this have a PIMPL?
    bool has_pimpl_() const noexcept;

    /// Object actually implementing *this
    pimpl_pointer m_pimpl_;
};

} // namespace tensorwrapper
