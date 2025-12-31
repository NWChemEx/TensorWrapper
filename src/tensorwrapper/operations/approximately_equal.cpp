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
#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::operations {
namespace {

struct Kernel {
    Kernel(double tolerance) : tol(tolerance) {}

    template<typename FloatType>
    bool operator()(const std::span<FloatType> result) {
        const FloatType zero{0.0};
        const FloatType ptol = static_cast<FloatType>(tol);
        for(std::size_t i = 0; i < result.size(); ++i) {
            auto diff = result[i];
            if(diff < zero) diff *= -1.0;
            if(diff >= ptol) return false;
        }
        return true;
    }

    double tol;
};

} // namespace

bool approximately_equal(const Tensor& lhs, const Tensor& rhs, double tol) {
    if(lhs.rank() != rhs.rank()) return false;

    std::string index(lhs.rank() ? "i0" : "");
    for(std::size_t i = 1; i < lhs.rank(); ++i)
        index += (",i" + std::to_string(i));
    Tensor result;
    result(index) = lhs(index) - rhs(index);

    using allocator_type = allocator::Contiguous;
    allocator_type alloc(result.buffer().allocator().runtime());
    const auto& buffer_down = alloc.rebind(result.buffer());
    Kernel k(tol);
    return wtf::buffer::visit_contiguous_buffer<types::floating_point_types>(
      k, buffer_down);
}

} // namespace tensorwrapper::operations
