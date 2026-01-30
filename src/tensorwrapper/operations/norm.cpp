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
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/operations/norm.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <wtf/buffer/float_buffer.hpp>

namespace tensorwrapper::operations {

Tensor infinity_norm(const Tensor& t) {
    const auto& buffer_down = buffer::make_contiguous(t.buffer());
    auto max_value          = buffer_down.infinity_norm();
    std::initializer_list<decltype(max_value)> il{max_value};
    using fp_types  = types::floating_point_types;
    auto wtf_buffer = wtf::buffer::make_float_buffer<fp_types>(il);
    shape::Smooth shape;
    buffer::Contiguous buffer(std::move(wtf_buffer), shape);
    layout::Physical playout(shape);
    layout::Logical llayout(shape);
    return Tensor(std::move(playout), std::move(llayout), std::move(buffer));
}

} // namespace tensorwrapper::operations
