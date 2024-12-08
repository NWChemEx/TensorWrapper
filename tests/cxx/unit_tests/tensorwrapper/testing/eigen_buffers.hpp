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

using ebufferd0 = buffer::Eigen<double, 0>;
using ebufferd1 = buffer::Eigen<double, 1>;

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
auto eigen_vector() {
    using buffer_type = buffer::Eigen<FloatType, 1>;
    using data_type   = typename buffer_type::data_type;
    data_type vector(5);
    for(std::size_t i = 0; i < 5; ++i) vector(i) = i;
    shape::Smooth shape{5};
    layout::Physical l(shape);
    return buffer_type(vector, l);
}

template<typename FloatType>
auto eigen_matrix() {
    using buffer_type = buffer::Eigen<FloatType, 2>;
    using data_type   = typename buffer_type::data_type;
    data_type matrix(2, 2);
    matrix(0, 0) = 1.0;
    matrix(0, 1) = 2.0;
    matrix(1, 0) = 3.0;
    matrix(1, 1) = 4.0;
    shape::Smooth shape{2, 2};
    layout::Physical l(shape);
    return buffer_type(matrix, l);
}

template<typename FloatType>
auto eigen_tensor3() {
    using buffer_type = buffer::Eigen<FloatType, 3>;
    using data_type   = typename buffer_type::data_type;
    shape::Smooth shape{2, 2, 2};
    layout::Physical l(shape);
    data_type tensor(2, 2, 2);
    tensor(0, 0, 0) = 1.0;
    tensor(0, 0, 1) = 2.0;
    tensor(0, 1, 0) = 3.0;
    tensor(0, 1, 1) = 4.0;
    tensor(1, 0, 0) = 5.0;
    tensor(1, 0, 1) = 6.0;
    tensor(1, 1, 0) = 7.0;
    tensor(1, 1, 1) = 8.0;
    return buffer_type(tensor, l);
}

} // namespace tensorwrapper::testing