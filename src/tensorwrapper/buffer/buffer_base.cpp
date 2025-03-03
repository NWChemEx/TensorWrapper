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

#include <tensorwrapper/buffer/buffer_base.hpp>

namespace tensorwrapper::buffer {

using dsl_reference = typename BufferBase::dsl_reference;

template<typename FxnType>
dsl_reference BufferBase::binary_op_common_(FxnType&& fxn,
                                            label_type this_labels,
                                            const_labeled_reference lhs,
                                            const_labeled_reference rhs) {
    const auto& lbuffer = lhs.object();
    const auto& rbuffer = rhs.object();

    auto llayout = lbuffer.layout()(lhs.labels());
    auto rlayout = rbuffer.layout()(rhs.labels());

    if(!has_layout()) m_layout_ = lbuffer.layout().clone_as<layout_type>();
    if(!has_allocator()) m_allocator_ = lbuffer.allocator().clone();

    fxn(m_layout_, this_labels, llayout, rlayout);

    return *this;
}

dsl_reference BufferBase::addition_assignment_(label_type this_labels,
                                               const_labeled_reference lhs,
                                               const_labeled_reference rhs) {
    auto lambda = [](auto&& obj, label_type this_labels, auto&& l, auto&& r) {
        obj->addition_assignment(this_labels, l, r);
    };

    return binary_op_common_(lambda, std::move(this_labels), lhs, rhs);
}

dsl_reference BufferBase::subtraction_assignment_(label_type this_labels,
                                                  const_labeled_reference lhs,
                                                  const_labeled_reference rhs) {
    auto lambda = [](auto&& obj, label_type this_labels, auto&& l, auto&& r) {
        obj->subtraction_assignment(this_labels, l, r);
    };

    return binary_op_common_(lambda, std::move(this_labels), lhs, rhs);
}

dsl_reference BufferBase::multiplication_assignment_(
  label_type this_labels, const_labeled_reference lhs,
  const_labeled_reference rhs) {
    auto lambda = [](auto&& obj, label_type this_labels, auto&& l, auto&& r) {
        obj->multiplication_assignment(this_labels, l, r);
    };

    return binary_op_common_(lambda, std::move(this_labels), lhs, rhs);
}

dsl_reference BufferBase::permute_assignment_(label_type this_labels,
                                              const_labeled_reference rhs) {
    auto rlayout = rhs.object().layout()(rhs.labels());

    if(!has_layout()) m_layout_ = rhs.object().layout().clone_as<layout_type>();
    if(!has_allocator()) m_allocator_ = rhs.object().allocator().clone();

    m_layout_->permute_assignment(this_labels, rlayout);

    return *this;
}

} // namespace tensorwrapper::buffer