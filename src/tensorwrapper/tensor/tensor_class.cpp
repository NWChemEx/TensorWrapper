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

#include "detail_/tensor_factory.hpp"
#include "detail_/tensor_pimpl.hpp"
#include <tensorwrapper/tensor/tensor_class.hpp>

namespace tensorwrapper {

using const_logical_reference = typename Tensor::const_logical_reference;
using const_buffer_reference  = typename Tensor::const_buffer_reference;

// -- Ctors, assignment, and dtor

Tensor::Tensor(detail_::TensorInput input) :
  Tensor(detail_::TensorFactory::construct(std::move(input))) {}

Tensor::Tensor(scalar_il_type il) :
  Tensor(detail_::TensorFactory::construct(il)) {}

Tensor::Tensor(vector_il_type il) :
  Tensor(detail_::TensorFactory::construct(il)) {}

Tensor::Tensor(matrix_il_type il) :
  Tensor(detail_::TensorFactory::construct(il)) {}

Tensor::Tensor(tensor3_il_type il) :
  Tensor(detail_::TensorFactory::construct(il)) {}

Tensor::Tensor(tensor4_il_type il) :
  Tensor(detail_::TensorFactory::construct(il)) {}

Tensor::Tensor(const Tensor& other) :
  m_pimpl_(other.has_pimpl_() ? other.m_pimpl_->clone() : nullptr) {}

Tensor::Tensor(Tensor&& other) noexcept = default;

Tensor& Tensor::operator=(const Tensor& rhs) {
    if(this != &rhs) Tensor(rhs).swap(*this);
    return *this;
}

Tensor& Tensor::operator=(Tensor&& rhs) noexcept = default;

Tensor::~Tensor() noexcept = default;

// -- Accessors

const_logical_reference Tensor::logical_layout() const {
    assert_pimpl_();
    return m_pimpl_->logical_layout();
}

const_buffer_reference Tensor::buffer() const {
    assert_pimpl_();
    return m_pimpl_->buffer();
}

// -- Utility

void Tensor::swap(Tensor& other) noexcept { m_pimpl_.swap(other.m_pimpl_); }

bool Tensor::operator==(const Tensor& rhs) const noexcept {
    if(has_pimpl_() != rhs.has_pimpl_()) return false;
    if(!has_pimpl_()) return true; // Both don't have a PIMPL
    return (*m_pimpl_) == (*rhs.m_pimpl_);
}

bool Tensor::operator!=(const Tensor& rhs) const noexcept {
    return !(*this == rhs);
}

// -- Private methods

Tensor::Tensor(pimpl_pointer pimpl) noexcept : m_pimpl_(std::move(pimpl)) {}

bool Tensor::has_pimpl_() const noexcept { return static_cast<bool>(m_pimpl_); }

void Tensor::assert_pimpl_() const {
    if(has_pimpl_()) return;
    throw std::runtime_error(
      "Tensor has no PIMPL. Was it default constructed?");
}

} // namespace tensorwrapper
