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
#include <tensorwrapper/tensor/detail_/tensor_input.hpp>

namespace tensorwrapper {
namespace detail_ {
class TensorPIMPL;
}

class Tensor {
private:
    /// Type of a helper class which collects the inputs needed to make a tensor
    using input_type = detail_::TensorInput;

public:
    /// Type of the object implementing *this
    using pimpl_type = detail_::TensorPIMPL;

    /// Type of a pointer to an object of type pimpl_type
    using pimpl_pointer = std::unique_ptr<pimpl_type>;

    /// Type of a read-only reference to the tensor's logical layout
    using const_logical_reference = input_type::const_logical_reference;

    /// Type of a pointer to the tensor's logical layout
    using logical_layout_pointer = input_type::logical_layout_pointer;

    /// Type of a read-only reference to the tensor's buffer
    using const_buffer_reference = input_type::const_buffer_reference;

    /// Type of a pointer to the tensor's buffer
    using buffer_pointer = input_type::buffer_pointer;

    template<typename... Args>
    Tensor(Args... args) : Tensor(input_type(std::forward<Args>(args)...)) {}

    /// Defaulted no-throw dtor
    ~Tensor() noexcept;

    const_logical_reference logical_layout() const;

    const_buffer_reference buffer() const;

private:
    explicit Tensor(input_type input);

    Tensor(pimpl_pointer pimpl) noexcept;

    /// Does *this have a PIMPL?
    bool has_pimpl_() const noexcept;

    /// Throws if *this does not have a PIMPL.
    void assert_pimpl_() const;

    /// Object actually implementing *this
    pimpl_pointer m_pimpl_;
};

} // namespace tensorwrapper
