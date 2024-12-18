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

typename BufferBase::labeled_buffer_type BufferBase::operator()(
  label_type labels) {
    return labeled_buffer_type(*this, std::move(labels));
}

typename BufferBase::labeled_const_buffer_type BufferBase::operator()(
  label_type labels) const {
    return labeled_const_buffer_type(*this, std::move(labels));
}

} // namespace tensorwrapper::buffer