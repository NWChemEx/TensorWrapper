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
#include "detail_/hash_utilities.hpp"
#include <tensorwrapper/buffer/mdbuffer.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::buffer {

MDBuffer::MDBuffer() noexcept = default;

MDBuffer::MDBuffer(buffer_type buffer, layout_pointer playout,
                   allocator_base_pointer pallocator) :
  my_base_type(std::move(playout), std::move(pallocator)),
  m_buffer_(std::move(buffer)) {}

// -----------------------------------------------------------------------------
// -- State Accessor
// -----------------------------------------------------------------------------

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

auto MDBuffer::shape_() const -> const_shape_view {
    return this->layout().shape().as_smooth();
}

auto MDBuffer::clone_() const -> buffer_base_pointer {
    return std::make_unique<MDBuffer>(*this);
}

bool MDBuffer::are_equal_(const_buffer_base_reference rhs) const noexcept {
    return my_base_type::template are_equal_impl_<my_type>(rhs);
}

auto MDBuffer::addition_assignment_(label_type this_labels,
                                    const_labeled_reference lhs,
                                    const_labeled_reference rhs)
  -> dsl_reference {}

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

auto MDBuffer::to_string_() const -> string_type {}

std::ostream& MDBuffer::add_to_stream_(std::ostream& os) const {}

// -----------------------------------------------------------------------------
// -- Private Methods
// -----------------------------------------------------------------------------

auto MDBuffer::coordinate_to_ordinal_(index_vector index) const -> size_type {
    using size_type   = typename decltype(index)::size_type;
    size_type ordinal = 0;
    size_type stride  = 1;
    for(rank_type i = shape_().rank(); i-- > 0;) {
        ordinal += index[i] * stride;
        stride *= shape_().extent(i);
    }
    return ordinal;
}

void MDBuffer::update_hash_() const {
    // for(auto i = 0; i < m_buffer_.size(); ++i)
    //     hash_utilities::hash_input(m_hash_, m_tensor_.data()[i]);
    m_recalculate_hash_ = false;
}

} // namespace tensorwrapper::buffer
