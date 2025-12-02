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

#include "../backends/eigen/eigen_tensor_impl.hpp"
#include "detail_/binary_operation_visitor.hpp"
#include "detail_/hash_utilities.hpp"
#include <tensorwrapper/buffer/mdbuffer.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::buffer {
namespace {

template<typename T>
const MDBuffer& downcast(T&& object) {
    auto* pobject = dynamic_cast<const MDBuffer*>(&object);
    if(pobject == nullptr) {
        throw std::invalid_argument("The provided buffer must be an MDBuffer.");
    }
    return *pobject;
}
} // namespace

using fp_types = types::floating_point_types;

MDBuffer::MDBuffer() noexcept = default;

MDBuffer::MDBuffer(buffer_type buffer, shape_type shape) :
  my_base_type(std::make_unique<layout::Physical>(shape), nullptr),
  m_shape_(std::move(shape)),
  m_buffer_() {
    if(buffer.size() == shape.size()) {
        m_buffer_ = std::move(buffer);
    } else {
        throw std::invalid_argument(
          "The size of the provided buffer does not match the size "
          "implied by the provided shape.");
    }
}

// -----------------------------------------------------------------------------
// -- State Accessor
// -----------------------------------------------------------------------------

auto MDBuffer::shape() const -> const_shape_view { return m_shape_; }

auto MDBuffer::size() const noexcept -> size_type { return m_buffer_.size(); }

auto MDBuffer::get_elem(index_vector index) const -> const_reference {
    auto ordinal_index = coordinate_to_ordinal_(index);
    return m_buffer_.at(ordinal_index);
}

void MDBuffer::set_elem(index_vector index, value_type new_value) {
    auto ordinal_index = coordinate_to_ordinal_(index);
    mark_for_rehash_();
    m_buffer_.at(ordinal_index) = new_value;
}

auto MDBuffer::get_mutable_data() -> buffer_view {
    mark_for_rehash_();
    return m_buffer_;
}

auto MDBuffer::get_immutable_data() const -> const_buffer_view {
    return m_buffer_;
}

// -----------------------------------------------------------------------------
// -- Utility Methods
// -----------------------------------------------------------------------------

bool MDBuffer::operator==(const my_type& rhs) const noexcept {
    if(!my_base_type::operator==(rhs)) return false;
    return get_hash_() == rhs.get_hash_();
}

// -----------------------------------------------------------------------------
// -- Protected Methods
// -----------------------------------------------------------------------------

auto MDBuffer::clone_() const -> buffer_base_pointer {
    return std::make_unique<MDBuffer>(*this);
}

bool MDBuffer::are_equal_(const_buffer_base_reference rhs) const noexcept {
    return my_base_type::template are_equal_impl_<my_type>(rhs);
}

auto MDBuffer::addition_assignment_(label_type this_labels,
                                    const_labeled_reference lhs,
                                    const_labeled_reference rhs)
  -> dsl_reference {
    const auto& lhs_down   = downcast(lhs.object());
    const auto& rhs_down   = downcast(rhs.object());
    const auto& lhs_labels = lhs.labels();
    const auto& rhs_labels = rhs.labels();
    const auto& lhs_shape  = lhs_down.m_shape_;
    const auto& rhs_shape  = rhs_down.m_shape_;

    auto labeled_lhs_shape = lhs_shape(lhs_labels);
    auto labeled_rhs_shape = rhs_shape(rhs_labels);

    m_shape_.addition_assignment(this_labels, labeled_lhs_shape,
                                 labeled_rhs_shape);

    detail_::AdditionVisitor visitor(m_buffer_, this_labels, m_shape_,
                                     lhs.labels(), lhs_shape, rhs.labels(),
                                     rhs_shape);

    wtf::buffer::visit_contiguous_buffer<fp_types>(visitor, lhs_down.m_buffer_,
                                                   rhs_down.m_buffer_);
    mark_for_rehash_();
    return *this;
}

auto MDBuffer::subtraction_assignment_(label_type this_labels,
                                       const_labeled_reference lhs,
                                       const_labeled_reference rhs)
  -> dsl_reference {}
auto MDBuffer::multiplication_assignment_(label_type this_labels,
                                          const_labeled_reference lhs,
                                          const_labeled_reference rhs)
  -> dsl_reference {}

auto MDBuffer::permute_assignment_(label_type this_labels,
                                   const_labeled_reference rhs)
  -> dsl_reference {}

auto MDBuffer::scalar_multiplication_(label_type this_labels, double scalar,
                                      const_labeled_reference rhs)
  -> dsl_reference {}

auto MDBuffer::to_string_() const -> string_type {
    std::stringstream ss;
    add_to_stream_(ss);
    return ss.str();
}

std::ostream& MDBuffer::add_to_stream_(std::ostream& os) const {
    /// XXX: EigenTensor should handle aliasing a const buffer correctly. That's
    ///      a lot of work, just to get this to work though...

    if(m_buffer_.size() == 0) return os;
    auto lambda = [&](auto&& span) {
        using clean_type = std::decay_t<decltype(span)>::value_type;
        auto data_ptr    = const_cast<clean_type*>(span.data());
        std::span<clean_type> data_span(data_ptr, span.size());
        auto ptensor = backends::eigen::make_eigen_tensor(data_span, m_shape_);
        ptensor->add_to_stream(os);
    };
    wtf::buffer::visit_contiguous_buffer<fp_types>(lambda, m_buffer_);
    return os;
}

// -----------------------------------------------------------------------------
// -- Private Methods
// -----------------------------------------------------------------------------

void MDBuffer::check_index_(const index_vector& index) const {
    if(index.size() != m_shape_.rank()) {
        throw std::out_of_range(
          "The length of the provided index does not match the rank of "
          "*this.");
    }
    for(rank_type i = 0; i < m_shape_.rank(); ++i) {
        if(index[i] >= m_shape_.extent(i)) {
            throw std::out_of_range(
              "An index provided is out of bounds for the corresponding "
              "dimension.");
        }
    }
}

auto MDBuffer::coordinate_to_ordinal_(index_vector index) const -> size_type {
    check_index_(index);
    using size_type   = typename decltype(index)::size_type;
    size_type ordinal = 0;
    size_type stride  = 1;
    for(rank_type i = shape().rank(); i-- > 0;) {
        ordinal += index[i] * stride;
        stride *= shape().extent(i);
    }
    return ordinal;
}

void MDBuffer::update_hash_() const {
    buffer::detail_::hash_utilities::HashVisitor visitor;
    if(m_buffer_.size()) {
        wtf::buffer::visit_contiguous_buffer<fp_types>(visitor, m_buffer_);
        m_hash_ = visitor.get_hash();
    }
    m_recalculate_hash_ = false;
}

} // namespace tensorwrapper::buffer
