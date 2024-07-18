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

#include "detail_/tensor_pimpl.hpp"
#include <tensorwrapper/tensor/tensor_class.hpp>

namespace tensorwrapper {

// -- Ctors, assignment, and dtor

Tensor::~Tensor() noexcept = default;

// -- Private methods

Tensor::Tensor(pimpl_pointer pimpl) noexcept : m_pimpl_(std::move(pimpl)) {}

bool Tensor::has_pimpl_() const noexcept { return static_cast<bool>(m_pimpl_); }

} // namespace tensorwrapper
