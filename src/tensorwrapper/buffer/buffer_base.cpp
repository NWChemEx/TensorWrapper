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

dsl_reference BufferBase::addition_assignment(label_type this_labels,
                                              const_labeled_reference rhs) {
    const auto& rlayout = rhs.object().layout();
    if(has_layout())
        m_layout_->addition_assignment(this_labels, rlayout(rhs.labels()));
    else
        throw std::runtime_error("For += result must be initialized");

    return addition_assignment_(std::move(this_labels), rhs);
}

dsl_reference BufferBase::permute_assignment(label_type this_labels,
                                             const_labeled_reference rhs) {
    const auto& rlayout = rhs.object().layout();
    if(has_layout())
        m_layout_->permute_assignment(this_labels, rlayout(rhs.labels()));
    else
        m_layout_ = rlayout.permute(rhs.labels(), this_labels);

    return permute_assignment_(std::move(this_labels), rhs);
}

} // namespace tensorwrapper::buffer
