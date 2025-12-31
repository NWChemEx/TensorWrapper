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

#include "../tensor/detail_/il_utils.hpp"
#include <tensorwrapper/allocator/contiguous.hpp>
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/detail_/unique_ptr_utilities.hpp>
#include <tensorwrapper/shape/smooth.hpp>

namespace tensorwrapper::allocator {

bool Contiguous::can_rebind(const_buffer_base_reference buffer) {
    auto pbuffer = dynamic_cast<const buffer_type*>(&buffer);
    return pbuffer != nullptr;
}

auto Contiguous::rebind(buffer_base_reference buffer) -> buffer_reference {
    if(can_rebind(buffer)) return static_cast<buffer_reference>(buffer);
    throw std::runtime_error("Can not rebind buffer");
}

auto Contiguous::rebind(const_buffer_base_reference buffer)
  -> const_buffer_reference {
    if(can_rebind(buffer)) return dynamic_cast<const_buffer_reference>(buffer);
    throw std::runtime_error("Can not rebind buffer");
}

// -----------------------------------------------------------------------------
// -- Protected methods
// -----------------------------------------------------------------------------

auto Contiguous::allocate_(layout_pointer playout) {
    return std::make_unique<buffer_type>(std::move(playout));
}

auto Contiguous::construct_(layout_pointer playout, wtf::fp::Float value)
  -> contiguous_pointer {
    auto pbuffer        = this->allocate(std::move(playout));
    auto& contig_buffer = static_cast<buffer::Contiguous&>(*pbuffer);
    contig_buffer.fill(value);
    return pbuffer;
}

// -- Private

auto Contiguous::layout_from_extents_(const std::vector<size_type>& extents)
  -> layout_pointer {
    shape::Smooth shape(extents.begin(), extents.end());
    return std::make_unique<layout::Physical>(std::move(shape));
}

} // namespace tensorwrapper::allocator
