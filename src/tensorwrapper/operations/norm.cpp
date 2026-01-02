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
#include <tensorwrapper/shape/smooth.hpp>
#include <tensorwrapper/tensor/tensor.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::operations {
namespace {
struct InfinityKernel {
    template<typename FloatType>
    auto operator()(const std::span<FloatType> buffer) {
        FloatType max_element{0.0};
        for(std::size_t i = 0; i < buffer.size(); ++i) {
            auto elem = types::fabs(buffer[i]);
            if(elem > max_element) max_element = elem;
        }
        shape::Smooth s{};
        std::vector<FloatType> data{max_element};
        auto pbuffer = std::make_unique<buffer::Contiguous>(data, s);
        return Tensor(s, std::move(pbuffer));
    }
};
} // namespace

Tensor infinity_norm(const Tensor& t) {
    // const auto& buffer_down = make_contiguous(t.buffer());
    //  InfinityKernel kernel;
    throw std::runtime_error("Fix me!!!!");
    // return wtf::buffer::visit_contiguous_buffer<types::floating_point_types>(
    //   kernel, buffer_down.get_immutable_data());
}

} // namespace tensorwrapper::operations
