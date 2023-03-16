/*
 * Copyright 2022 NWChemEx-Project
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

#include "detail_/buffer_pimpl.hpp"
#include "tensorwrapper/tensor/buffer/buffer.hpp"

namespace tensorwrapper::tensor::buffer {

#define TEMPLATE_PARAMS template<typename FieldType>
#define BUFFER Buffer<FieldType>

TEMPLATE_PARAMS
BUFFER::Buffer() noexcept = default;

TEMPLATE_PARAMS
BUFFER::Buffer(pimpl_pointer pimpl) noexcept : m_pimpl_(std::move(pimpl)) {}

TEMPLATE_PARAMS
BUFFER::Buffer(const Buffer& other) :
  m_pimpl_(other.m_pimpl_ ? other.m_pimpl_->clone() : nullptr) {}

TEMPLATE_PARAMS
BUFFER::Buffer(Buffer&& other) noexcept : m_pimpl_(std::move(other.m_pimpl_)) {}

TEMPLATE_PARAMS
BUFFER& BUFFER::operator=(const Buffer& rhs) {
    if(rhs.m_pimpl_) {
        rhs.m_pimpl_->clone().swap(m_pimpl_);
    } else {
        m_pimpl_.reset();
    }
    return *this;
}

TEMPLATE_PARAMS
BUFFER& BUFFER::operator=(Buffer&& rhs) noexcept = default;

TEMPLATE_PARAMS
BUFFER::~Buffer() noexcept = default;

TEMPLATE_PARAMS
typename BUFFER::pimpl_type* BUFFER::pimpl() noexcept { return m_pimpl_.get(); }

TEMPLATE_PARAMS
const typename BUFFER::pimpl_type* BUFFER::pimpl() const noexcept {
    return m_pimpl_.get();
}

TEMPLATE_PARAMS
void BUFFER::swap(Buffer& rhs) noexcept { std::swap(m_pimpl_, rhs.m_pimpl_); }

TEMPLATE_PARAMS
bool BUFFER::is_initialized() const noexcept {
    return static_cast<bool>(m_pimpl_);
}

TEMPLATE_PARAMS
void BUFFER::permute(const_annotation_reference my_idx,
                     const_annotation_reference out_idx, my_type& out) const {
    assert_initialized_();
    if(!out.is_initialized()) default_initialize_(out);
    m_pimpl_->permute(my_idx, out_idx, *out.m_pimpl_);
}

TEMPLATE_PARAMS
void BUFFER::scale(const_annotation_reference my_idx,
                   const_annotation_reference out_idx, my_type& out,
                   double rhs) const {
    assert_initialized_();
    if(!out.is_initialized()) default_initialize_(out);
    m_pimpl_->scale(my_idx, out_idx, *out.m_pimpl_, rhs);
}

TEMPLATE_PARAMS
void BUFFER::add(const_annotation_reference my_idx,
                 const_annotation_reference out_idx, my_type& out,
                 const_annotation_reference rhs_idx, const my_type& rhs) const {
    assert_initialized_();
    rhs.assert_initialized_();
    if(!out.is_initialized()) default_initialize_(out);
    m_pimpl_->add(my_idx, out_idx, *out.m_pimpl_, rhs_idx, *rhs.m_pimpl_);
}

TEMPLATE_PARAMS
void BUFFER::inplace_add(const_annotation_reference my_idx,
                         const_annotation_reference rhs_idx,
                         const my_type& rhs) {
    assert_initialized_();
    rhs.assert_initialized_();
    m_pimpl_->inplace_add(my_idx, rhs_idx, *rhs.m_pimpl_);
}

TEMPLATE_PARAMS
void BUFFER::subtract(const_annotation_reference my_idx,
                      const_annotation_reference out_idx, my_type& out,
                      const_annotation_reference rhs_idx,
                      const my_type& rhs) const {
    assert_initialized_();
    rhs.assert_initialized_();
    if(!out.is_initialized()) default_initialize_(out);
    m_pimpl_->subtract(my_idx, out_idx, *out.m_pimpl_, rhs_idx, *rhs.m_pimpl_);
}

TEMPLATE_PARAMS
void BUFFER::inplace_subtract(const_annotation_reference my_idx,
                              const_annotation_reference rhs_idx,
                              const my_type& rhs) {
    assert_initialized_();
    rhs.assert_initialized_();
    m_pimpl_->inplace_subtract(my_idx, rhs_idx, *rhs.m_pimpl_);
}

TEMPLATE_PARAMS
void BUFFER::times(const_annotation_reference my_idx,
                   const_annotation_reference out_idx, my_type& out,
                   const_annotation_reference rhs_idx,
                   const my_type& rhs) const {
    assert_initialized_();
    rhs.assert_initialized_();
    if(!out.is_initialized()) default_initialize_(out);
    m_pimpl_->times(my_idx, out_idx, *out.m_pimpl_, rhs_idx, *rhs.m_pimpl_);
}

TEMPLATE_PARAMS
typename BUFFER::scalar_value_type BUFFER::dot(
  const_annotation_reference my_idx, const_annotation_reference rhs_idx,
  const my_type& rhs) const {
    assert_initialized_();
    rhs.assert_initialized_();
    return m_pimpl_->dot(my_idx, rhs_idx, *rhs.m_pimpl_);
}

TEMPLATE_PARAMS
typename BUFFER::scalar_value_type BUFFER::norm() const {
    assert_initialized_();
    return m_pimpl_->norm();
}

TEMPLATE_PARAMS
typename BUFFER::scalar_value_type BUFFER::sum() const {
    assert_initialized_();
    return m_pimpl_->sum();
}

TEMPLATE_PARAMS
typename BUFFER::scalar_value_type BUFFER::trace() const {
    assert_initialized_();
    return m_pimpl_->trace();
}

TEMPLATE_PARAMS
typename BUFFER::extents_type BUFFER::make_extents() const {
    assert_initialized_();
    return m_pimpl_->make_extents();
}

TEMPLATE_PARAMS
typename BUFFER::inner_extents_type BUFFER::make_inner_extents() const {
    assert_initialized_();
    return m_pimpl_->make_inner_extents();
}

TEMPLATE_PARAMS
bool BUFFER::operator==(const Buffer& rhs) const noexcept {
    if(m_pimpl_ && rhs.m_pimpl_) {
        return m_pimpl_->are_equal(*rhs.m_pimpl_);
    } else if(!m_pimpl_ && !rhs.m_pimpl_)
        return true;
    return false;
}

TEMPLATE_PARAMS
std::ostream& BUFFER::print(std::ostream& os) const {
    if(!is_initialized()) return os;
    return os << *m_pimpl_;
}

// -- Private methods ----------------------------------------------------------

TEMPLATE_PARAMS
void BUFFER::assert_initialized_() const {
    if(is_initialized()) return;
    throw std::runtime_error("Buffer instance currently does not wrap a value. "
                             "Did you forget to initialize it?");
}

TEMPLATE_PARAMS
void BUFFER::default_initialize_(Buffer& other) const {
    Buffer(m_pimpl_->default_clone()).swap(other);
}

#undef BUFFER
#undef TEMPLATE_PARAMS

template class Buffer<field::Scalar>;
template class Buffer<field::Tensor>;

} // namespace tensorwrapper::tensor::buffer
