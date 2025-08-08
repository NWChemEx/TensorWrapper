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

#include <tensorwrapper/allocator/eigen.hpp>
#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/utilities/floating_point_dispatch.hpp>

namespace tensorwrapper::operations {
namespace {

struct Kernel {
    template<typename FloatType>
    bool run(const buffer::BufferBase& result, double tol) {
        using allocator_type = allocator::Eigen<FloatType>;
        const FloatType zero{0.0};
        const FloatType ptol = static_cast<FloatType>(tol);
        auto& buffer_down    = allocator_type::rebind(result);

        for(std::size_t i = 0; i < buffer_down.size(); ++i) {
            auto diff = buffer_down.get_data(i);
            if(diff < zero) diff *= -1.0;
            if(diff >= ptol) return false;
        }
        return true;
    }
};

} // namespace

bool approximately_equal(const Tensor& lhs, const Tensor& rhs, double tol) {
    if(lhs.rank() != rhs.rank()) return false;

    std::string index(lhs.rank() ? "i0" : "");
    for(std::size_t i = 1; i < lhs.rank(); ++i)
        index += (",i" + std::to_string(i));
    Tensor result;
    result(index) = lhs(index) - rhs(index);

    using tensorwrapper::utilities::floating_point_dispatch;

    return floating_point_dispatch(Kernel{}, result.buffer(), tol);
}

} // namespace tensorwrapper::operations
