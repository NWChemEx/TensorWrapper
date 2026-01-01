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

#pragma once
#include <tensorwrapper/tensorwrapper.hpp>

/** @file eigen_buffers.hpp
 *
 *  This file creates some hard-coded buffer::Eigen objects that can be used
 *  for testing.
 *
 */

namespace tensorwrapper::testing {

template<typename FloatType>
auto eigen_scalar(FloatType value = 42.0) {
    shape::Smooth shape{};
    std::vector<FloatType> data{value};
    return std::make_unique<buffer::Contiguous>(std::move(data),
                                                std::move(shape));
}

template<typename FloatType>
auto eigen_vector(std::size_t n = 5) {
    shape::Smooth shape{n};
    std::vector<FloatType> data(n);
    for(std::size_t i = 0; i < n; ++i) data[i] = static_cast<FloatType>(i);
    return std::make_unique<buffer::Contiguous>(std::move(data),
                                                std::move(shape));
}

template<typename FloatType>
auto eigen_matrix(std::size_t n = 2, std::size_t m = 2) {
    shape::Smooth shape{n, m};
    std::vector<FloatType> data(n * m);
    double counter = 1.0;
    for(decltype(n) i = 0; i < n; ++i)
        for(decltype(m) j = 0; j < m; ++j)
            data[i * m + j] = static_cast<FloatType>(counter++);
    return std::make_unique<buffer::Contiguous>(std::move(data),
                                                std::move(shape));
}

template<typename FloatType>
auto eigen_tensor3(std::size_t n = 2, std::size_t m = 2, std::size_t l = 2) {
    shape::Smooth shape{n, m, l};
    std::vector<FloatType> data(n * m * l);
    double counter = 1.0;
    for(decltype(n) i = 0; i < n; ++i)
        for(decltype(m) j = 0; j < m; ++j)
            for(decltype(l) k = 0; k < l; ++k)
                data[i * m * n + j * n + l] = static_cast<FloatType>(counter++);
    return std::make_unique<buffer::Contiguous>(std::move(data),
                                                std::move(shape));
}

template<typename FloatType>
auto eigen_tensor4(std::array<std::size_t, 4> extents = {2, 2, 2, 2}) {
    shape::Smooth shape(extents.begin(), extents.end());
    std::vector<FloatType> data(shape.size());
    buffer::Contiguous buffer(std::move(data), std::move(shape));
    double counter = 1.0;
    for(std::size_t i = 0; i < extents[0]; ++i)
        for(decltype(i) j = 0; j < extents[1]; ++j)
            for(decltype(i) k = 0; k < extents[2]; ++k)
                for(decltype(i) l = 0; l < extents[3]; ++l)
                    buffer.set_elem({i, j, k, l},
                                    static_cast<FloatType>(counter++));

    return std::make_unique<buffer::Contiguous>(std::move(buffer));
}

} // namespace tensorwrapper::testing
