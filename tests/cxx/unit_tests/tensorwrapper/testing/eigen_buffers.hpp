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

// Typedefs of buffer::Eigen objects with various template parameters
using ebufferf0 = buffer::Eigen<float, 0>;
using ebufferf1 = buffer::Eigen<float, 1>;
using ebufferf2 = buffer::Eigen<float, 2>;
using ebufferf3 = buffer::Eigen<float, 3>;
using ebufferd0 = buffer::Eigen<double, 0>;
using ebufferd1 = buffer::Eigen<double, 1>;
using ebufferd2 = buffer::Eigen<double, 2>;
using ebufferd3 = buffer::Eigen<double, 3>;

template<typename FloatType>
auto eigen_scalar() {
    using buffer_type = buffer::Eigen<FloatType, 0>;
    using data_type   = typename buffer_type::data_type;
    data_type scalar;
    scalar() = 42.0;
    shape::Smooth shape{};
    layout::Physical l(shape);
    return buffer_type(scalar, l);
}

template<typename FloatType>
auto eigen_vector(std::size_t n = 5) {
    using buffer_type = buffer::Eigen<FloatType, 1>;
    using data_type   = typename buffer_type::data_type;
    data_type vector(n);
    for(std::size_t i = 0; i < n; ++i) vector(i) = i;
    shape::Smooth shape{n};
    layout::Physical l(shape);
    return buffer_type(vector, l);
}

template<typename FloatType>
auto eigen_matrix(std::size_t n = 2, std::size_t m = 2) {
    using buffer_type = buffer::Eigen<FloatType, 2>;
    using data_type   = typename buffer_type::data_type;
    data_type matrix(n, m);
    double counter = 1.0;
    for(std::size_t i = 0; i < n; ++i)
        for(std::size_t j = 0; j < m; ++j) matrix(i, j) = counter++;

    shape::Smooth shape{n, m};
    layout::Physical l(shape);
    return buffer_type(matrix, l);
}

template<typename FloatType>
auto eigen_tensor3(std::size_t n = 2, std::size_t m = 2, std::size_t l = 2) {
    using buffer_type = buffer::Eigen<FloatType, 3>;
    using data_type   = typename buffer_type::data_type;
    int in            = static_cast<int>(n);
    int im            = static_cast<int>(m);
    int il            = static_cast<int>(l);
    shape::Smooth shape{n, m, l};
    layout::Physical layout(shape);
    data_type e_tensor(in, im, il);
    double counter = 1.0;
    for(decltype(in) i = 0; i < in; ++i)
        for(decltype(im) j = 0; j < im; ++j)
            for(decltype(il) k = 0; k < il; ++k) e_tensor(i, j, k) = counter++;
    return buffer_type(e_tensor, layout);
}

template<typename FloatType>
auto eigen_tensor4(std::array<std::size_t, 4> extents = {2, 2, 2, 2}) {
    auto constexpr Rank = 4;
    using buffer_type   = buffer::Eigen<FloatType, Rank>;
    using data_type     = typename buffer_type::data_type;
    std::array<int, Rank> iextents;
    for(std::size_t i = 0; i < Rank; ++i) iextents[i] = extents[i];
    shape::Smooth shape{extents[0], extents[1], extents[2], extents[3]};
    layout::Physical layout(shape);
    data_type e_tensor(iextents[0], iextents[1], iextents[2], iextents[3]);
    double counter = 1.0;
    std::array<int, Rank> i;
    for(i[0] = 0; i[0] < iextents[0]; ++i[0])
        for(i[1] = 0; i[1] < iextents[1]; ++i[1])
            for(i[2] = 0; i[2] < iextents[2]; ++i[2])
                for(i[3] = 0; i[3] < iextents[3]; ++i[3])
                    e_tensor(i[0], i[1], i[2], i[3]) = counter++;
    return buffer_type(e_tensor, layout);
}

} // namespace tensorwrapper::testing