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

#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/operations/power.hpp>
#include <tensorwrapper/types/floating_point.hpp>

namespace tensorwrapper::operations {
namespace {

class PowerKernel {
public:
    PowerKernel(double pow) : m_pow_(pow) {}

    template<typename FloatType>
    void operator()(std::span<FloatType> A) const {
        if constexpr(std::is_const_v<FloatType>) {
            // This path is only for the compiler, we won't actually get to
            // it.
            throw std::runtime_error("Can't modify const data");
        } else {
            for(auto& a : A) { a = types::pow(a, m_pow_); }
        }
    }

private:
    double m_pow_;
};
} // namespace

Tensor power(Tensor A, double pow) {
    PowerKernel kernel(pow);
    auto& buffer = make_contiguous(A.buffer());
    buffer::visit_contiguous_buffer(kernel, buffer);
    return A;
}
} // namespace tensorwrapper::operations
