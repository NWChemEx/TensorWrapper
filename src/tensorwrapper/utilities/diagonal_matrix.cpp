/*
 * Copyright 2026 NWChemEx-Project
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

#include <stdexcept>
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/utilities/diagonal_matrix.hpp>

namespace tensorwrapper::utilities {
namespace {
struct Kernel {
    template<typename FloatType>
    auto operator()(const std::span<FloatType>& diagonal_elements) {
        using clean_type = std::decay_t<FloatType>;
        const auto n     = diagonal_elements.size();
        shape::Smooth new_shape{n, n};
        std::vector<clean_type> data(n * n, 0);
        for(std::size_t i = 0; i < n; ++i) {
            data[i * n + i] = diagonal_elements[i];
        }
        buffer::Contiguous buffer(data, new_shape);
        return Tensor(std::move(new_shape), std::move(buffer));
    }
};
} // namespace

Tensor diagonal_matrix(const Tensor& diagonal_elements) {
    if(diagonal_elements.rank() != 1) {
        throw std::runtime_error("Diagonal elements must be a vector");
    }
    Kernel k;
    auto& buffer = make_contiguous(diagonal_elements.buffer());
    return buffer::visit_contiguous_buffer(k, buffer);
}

} // namespace tensorwrapper::utilities
