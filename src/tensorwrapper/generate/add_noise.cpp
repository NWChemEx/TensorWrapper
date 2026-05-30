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
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <vector>

namespace tensorwrapper::generate {
namespace {

std::vector<std::size_t> shape_extents(const auto& shape) {
    std::vector<std::size_t> rv(shape.rank());
    for(std::size_t d = 0; d < shape.rank(); ++d) rv[d] = shape.extent(d);
    return rv;
}

} // namespace

Tensor add_noise(const Tensor& matrix, double t, std::mt19937& gen) {
    if(t < 0.0) { throw std::invalid_argument("t must be non-negative"); }

    auto& in_buf = buffer::make_contiguous(matrix.buffer());
    auto in_data = buffer::get_raw_data<const double>(in_buf);
    std::vector<double> data(in_data.begin(), in_data.end());

    if(t > 0.0) {
        std::normal_distribution<double> dist(0.0, t);
        for(auto& x : data) { x += std::clamp(dist(gen), -t, t); }
    }

    return utilities::make_tensor(shape_extents(in_buf.shape()), data.begin(),
                                  data.end());
}

Tensor add_noise(const Tensor& matrix, double t, std::uint64_t seed) {
    auto gen = make_rng(seed);
    return add_noise(matrix, t, gen);
}

} // namespace tensorwrapper::generate
