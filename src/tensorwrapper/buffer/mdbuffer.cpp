/*
 * Copyright 2025 NWChemEx-Project
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

#include "detail_/addition_visitor.hpp"
#include "detail_/mdbuffer_pimpl.hpp"
#include <tensorwrapper/buffer/mdbuffer.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::buffer {

MDBuffer::MDBuffer() noexcept : m_pimpl_(nullptr) {}

MDBuffer::MDBuffer(shape_type shape, buffer_type buffer) :
  MDBuffer(std::make_unique<detail_::MDBufferPIMPL>(std::move(shape),
                                                    std::move(buffer))) {}

MDBuffer::MDBuffer(pimpl_pointer pimpl) noexcept : m_pimpl_(std::move(pimpl)) {}

auto MDBuffer::rank() const -> rank_type {
    assert_pimpl_();
    return m_pimpl_->shape().rank();
}

bool MDBuffer::has_pimpl_() const noexcept { return m_pimpl_ != nullptr; }

void MDBuffer::assert_pimpl_() const {
    if(!has_pimpl_()) {
        throw std::runtime_error(
          "MDBuffer has no PIMPL. Was it default constructed?");
    }
}

auto MDBuffer::pimpl_() -> pimpl_type& {
    assert_pimpl_();
    return *m_pimpl_;
}

auto MDBuffer::pimpl_() const -> const pimpl_type& {
    assert_pimpl_();
    return *m_pimpl_;
}

} // namespace tensorwrapper::buffer
