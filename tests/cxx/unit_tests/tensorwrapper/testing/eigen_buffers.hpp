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
auto make_allocator() {
    parallelzone::runtime::RuntimeView rv;
    return allocator::Eigen<FloatType>(rv);
}

template<typename FloatType>
auto eigen_scalar(FloatType value = 42.0) {
    auto alloc = make_allocator<FloatType>();
    return alloc.construct(value);
}

template<typename FloatType>
auto eigen_vector(std::size_t n = 5) {
    layout::Physical l(shape::Smooth{n});
    auto alloc  = make_allocator<FloatType>();
    auto buffer = alloc.allocate(l);
    for(std::size_t i = 0; i < n; ++i) buffer->set_elem({i}, i);
    return buffer;
}

template<typename FloatType>
auto eigen_matrix(std::size_t n = 2, std::size_t m = 2) {
    layout::Physical l(shape::Smooth{n, m});
    auto alloc     = make_allocator<FloatType>();
    auto buffer    = alloc.allocate(l);
    double counter = 1.0;
    for(decltype(n) i = 0; i < n; ++i)
        for(decltype(m) j = 0; j < m; ++j) buffer->set_elem({i, j}, counter++);
    return buffer;
}

template<typename FloatType>
auto eigen_tensor3(std::size_t n = 2, std::size_t m = 2, std::size_t l = 2) {
    layout::Physical layout(shape::Smooth{n, m, l});
    auto alloc     = make_allocator<FloatType>();
    auto buffer    = alloc.allocate(layout);
    double counter = 1.0;
    for(decltype(n) i = 0; i < n; ++i)
        for(decltype(m) j = 0; j < m; ++j)
            for(decltype(l) k = 0; k < l; ++k)
                buffer->set_elem({i, j, k}, counter++);
    return buffer;
}

template<typename FloatType>
auto eigen_tensor4(std::array<std::size_t, 4> extents = {2, 2, 2, 2}) {
    shape::Smooth shape{extents[0], extents[1], extents[2], extents[3]};
    layout::Physical layout(shape);
    auto alloc     = make_allocator<FloatType>();
    auto buffer    = alloc.allocate(layout);
    double counter = 1.0;
    decltype(extents) i;
    for(i[0] = 0; i[0] < extents[0]; ++i[0])
        for(i[1] = 0; i[1] < extents[1]; ++i[1])
            for(i[2] = 0; i[2] < extents[2]; ++i[2])
                for(i[3] = 0; i[3] < extents[3]; ++i[3])
                    buffer->set_elem({i[0], i[1], i[2], i[3]}, counter++);
    return buffer;
}

} // namespace tensorwrapper::testing