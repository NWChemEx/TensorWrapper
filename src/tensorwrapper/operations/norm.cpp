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
#include <tensorwrapper/allocator/contiguous.hpp>
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/tensor/tensor.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::operations {
namespace {
struct InfinityKernel {
    InfinityKernel(allocator::Contiguous& alloc) : palloc(&alloc) {}

    template<typename FloatType>
    auto operator()(const std::span<FloatType> buffer) {
        FloatType max_element{0.0};
        for(std::size_t i = 0; i < buffer.size(); ++i) {
            auto elem = types::fabs(buffer[i]);
            if(elem > max_element) max_element = elem;
        }
        shape::Smooth s{};
        layout::Physical l(s);
        auto pbuffer = palloc->construct(l, max_element);
        return Tensor(s, std::move(pbuffer));
    }

    allocator::Contiguous* palloc;
};

} // namespace

Tensor infinity_norm(const Tensor& t) {
    using allocator_type = allocator::Contiguous;
    auto rv              = t.buffer().allocator().runtime();
    allocator_type alloc(rv);
    const auto& buffer_down = alloc.rebind(t.buffer());
    InfinityKernel kernel(alloc);
    throw std::runtime_error("Fix me!!!!");
    // return wtf::buffer::visit_contiguous_buffer<types::floating_point_types>(
    //   kernel, buffer_down);
}

} // namespace tensorwrapper::operations
