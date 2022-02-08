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
void BUFFER::swap(Buffer& rhs) noexcept { std::swap(m_pimpl_, rhs.m_pimpl_); }

TEMPLATE_PARAMS
bool BUFFER::is_initialized() const noexcept {
    return static_cast<bool>(m_pimpl_);
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
bool BUFFER::operator==(const Buffer& rhs) const noexcept {
    if(m_pimpl_ && rhs.m_pimpl_) {
        return m_pimpl_->are_equal(*rhs.m_pimpl_);
    } else if(!m_pimpl_ && !rhs.m_pimpl_)
        return true;
    return false;
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
