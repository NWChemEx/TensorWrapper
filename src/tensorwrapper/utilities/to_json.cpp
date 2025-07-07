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

#include <tensorwrapper/buffer/eigen.hpp>
#include <tensorwrapper/utilities/to_json.hpp>

namespace tensorwrapper::utilities {

using offset_type   = std::size_t;
using offset_vector = std::vector<offset_type>;

template<typename DataType>
using buffer_type = buffer::Contiguous<DataType>;

template<typename DataType>
void to_json_(std::ostream& os, const buffer_type<DataType>& t,
              offset_vector index) {
    const auto& shape = t.layout().shape().as_smooth();
    auto rank         = index.size();
    if(rank == t.rank()) {
        os << t.get_elem(index);
        return;
    } else {
        auto n_elements = shape.extent(rank);
        index.push_back(0);
        os << '[';
        for(decltype(n_elements) i = 0; i < n_elements; ++i) {
            index[rank] = i;
            to_json_(os, t, index);
            if(i + 1 < n_elements) os << ',';
        }
        os << ']';
    }
}

std::ostream& to_json(std::ostream& os, const Tensor& t) {
    offset_vector i;
    const auto& buffer = buffer::to_eigen_buffer<double>(t.buffer());
    to_json_(os, buffer, i);
    return os;
}

} // namespace tensorwrapper::utilities