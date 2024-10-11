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
#include <deque>
#include <optional>
#include <tuple>

namespace {

/// Recursion end-point for unwrap_il
auto unwrap_il(double il) {
    return std::make_tuple(std::deque<int>{}, std::vector<double>{il});
}

/** Unwraps a (possibly) recursive initializer list filled with doubles
 *
 *  @return A pair such that the zeroth element is the shape of @p il and the
 *          first element is the data of @p il unrolled into nested vectors.
 */
template<typename T>
auto unwrap_il(std::initializer_list<T> il) {
    using return_types = decltype(unwrap_il(std::declval<T>()));

    std::optional<std::tuple_element_t<0, return_types>> rv_dims;
    std::vector<double> rv_data;

    for(auto b = il.begin(); b != il.end(); ++b) {
        auto [dims, data] = unwrap_il(*b);
        if(!rv_dims.has_value())
            rv_dims.emplace(dims);
        else {
            if(*rv_dims != dims) throw std::runtime_error("Not smooth");
        }
        for(auto x : data) rv_data.push_back(x);
    }
    if(rv_dims)
        rv_dims->push_front(il.size());
    else // If il is empty rv_dims never gets initialized
        rv_dims.emplace(std::deque<int>{});
    return std::make_tuple(*rv_dims, rv_data);
}

} // namespace