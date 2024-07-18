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
#include <tensorwrapper/tensor/detail_/tensor_factory.hpp>

namespace tensorwrapper {

class Tensor {
private:
    /// Type of factory associated with *this
    using factory_type = detail_::TensorFactory;

public:
    /// Type of the object implementing *this
    using pimpl_type = typename factory_type::pimpl_type;

    /// Type of a pointer to an object of type pimpl_type
    using pimpl_pointer = typename factory_type::pimpl_pointer;

    template<typename... Args>
    Tensor(Args... args) :
      Tensor(detail_::construct(std::forward<Args>(args)...)) {}

    /// Defaulted no-throw dtor
    ~Tensor() noexcept;

private:
    Tensor(pimpl_pointer pimpl) noexcept;

    /// Does *this have a PIMPL?
    bool has_pimpl_() const noexcept;

    /// Object actually implementing *this
    pimpl_pointer m_pimpl_;
};

} // namespace tensorwrapper
