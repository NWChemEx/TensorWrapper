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

#include <algorithm>
#include <random>
#include <stdexcept>
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/generate/add_noise.hpp>
#include <tensorwrapper/generate/generate_utils.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <vector>

namespace tensorwrapper::generate {
namespace {

std::vector<std::size_t> shape_extents(const auto& shape) {
    std::vector<std::size_t> rv(shape.rank());
    for(std::size_t d = 0; d < shape.rank(); ++d) rv[d] = shape.extent(d);
    return rv;
}

template<concepts::FloatingPoint T>
Tensor add_noise_impl(const Tensor& matrix, double t, std::mt19937& gen) {
    if(t < 0.0) { throw std::invalid_argument("t must be non-negative"); }

    auto& in_buf = buffer::make_contiguous(matrix.buffer());
    auto in_data = buffer::get_raw_data<const T>(in_buf);
    std::vector<T> data(in_data.begin(), in_data.end());

    if(t > 0.0) {
        std::normal_distribution<double> dist(0.0, t);
        for(auto& x : data) {
            const auto center = tensorwrapper::types::uq_center(x);
            const auto delta  = std::clamp(dist(gen), -t, t);
            x = tensorwrapper::types::construct_uq_type<T>(center, delta);
        }
    }

    return utilities::make_tensor(shape_extents(in_buf.shape()), data.begin(),
                                  data.end());
}

} // namespace

template<concepts::FloatingPoint T>
Tensor add_noise(const Tensor& matrix, double t, std::mt19937& gen) {
    return add_noise_impl<T>(matrix, t, gen);
}

template<concepts::FloatingPoint T>
Tensor add_noise(const Tensor& matrix, double t, std::uint64_t seed) {
    auto gen = make_rng(seed);
    return add_noise<T>(matrix, t, gen);
}

Tensor add_noise(const Tensor& matrix, double t, std::mt19937& gen) {
    return add_noise<double>(matrix, t, gen);
}

Tensor add_noise(const Tensor& matrix, double t, std::uint64_t seed) {
    return add_noise<double>(matrix, t, seed);
}

#define DEFINE_ADD_NOISE(TYPE)                                      \
    template Tensor add_noise<TYPE>(const Tensor& matrix, double t, \
                                    std::mt19937& gen);             \
    template Tensor add_noise<TYPE>(const Tensor& matrix, double t, \
                                    std::uint64_t seed);

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_ADD_NOISE);

#undef DEFINE_ADD_NOISE

} // namespace tensorwrapper::generate
