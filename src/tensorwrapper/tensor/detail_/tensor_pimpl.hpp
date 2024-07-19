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
#include <tensorwrapper/tensor/tensor_class.hpp>

namespace tensorwrapper::detail_ {

class TensorPIMPL {
public:
    /// Type *this implements
    using parent_type = Tensor;

    /// Pull in types from parent_type
    using pimpl_pointer          = typename parent_type::pimpl_pointer;
    using logical_layout_pointer = typename parent_type::logical_layout_pointer;
    using buffer_pointer         = typename parent_type::buffer_pointer;

    TensorPIMPL(logical_layout_pointer plogical, buffer_pointer pbuffer) :
      m_plogical_(std::move(plogical)), m_pbuffer_(std::move(pbuffer)) {}

    auto& logical_layout() { return *m_plogical_; }
    const auto& logical_layout() const { return *m_plogical_; }

    auto& buffer() { return *m_pbuffer_; }
    const auto& buffer() const { return *m_pbuffer_; }

private:
    logical_layout_pointer m_plogical_;

    buffer_pointer m_pbuffer_;
};

} // namespace tensorwrapper::detail_
