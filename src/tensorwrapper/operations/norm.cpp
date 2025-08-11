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

#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/tensor/tensor.hpp>
#include <tensorwrapper/utilities/floating_point_dispatch.hpp>

namespace tensorwrapper::operations {
namespace {
struct InfinityKernel {
    template<typename FloatType>
    Tensor run(const buffer::BufferBase& t) {
        using allocator_type = allocator::Eigen<FloatType>;
        allocator_type alloc(t.allocator().runtime());
        FloatType max_element{0.0};
        const auto& buffer_down = alloc.rebind(t);
        for(std::size_t i = 0; i < buffer_down.size(); ++i) {
            auto elem = types::fabs(buffer_down.get_data(i));
            if(elem > max_element) max_element = elem;
        }
        shape::Smooth s{};
        layout::Physical l(s);
        auto pbuffer = alloc.construct(l, max_element);
        return Tensor(s, std::move(pbuffer));
    }
};

} // namespace

Tensor infinity_norm(const Tensor& t) {
    InfinityKernel k;
    return utilities::floating_point_dispatch(k, t.buffer());
}

} // namespace tensorwrapper::operations
