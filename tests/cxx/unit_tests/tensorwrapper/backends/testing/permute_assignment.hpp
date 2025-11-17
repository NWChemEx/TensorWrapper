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

#pragma once
#include "unary_op.hpp"

namespace tensorwrapper::testing {

template<typename TestType>
void scalar_permute_assignment() {
    auto the_op = [](auto&& out_idx, auto&& rhs_idx, auto&& result, auto&& t0) {
        result.permute_assignment(out_idx, rhs_idx, t0);
    };
    auto corr_op = [](auto a) { return a; };
    scalar_unary_assignment<TestType>(the_op, corr_op);
}

template<typename TestType>
void vector_permute_assignment() {
    auto the_op = [](auto&& out_idx, auto&& rhs_idx, auto&& result, auto&& t0) {
        result.permute_assignment(out_idx, rhs_idx, t0);
    };
    auto corr_op = [](auto a) { return a; };
    vector_unary_assignment<TestType>(the_op, corr_op);
}

template<typename TestType>
void matrix_permute_assignment() {
    auto the_op = [](auto&& out_idx, auto&& rhs_idx, auto&& result, auto&& t0) {
        result.permute_assignment(out_idx, rhs_idx, t0);
    };
    auto corr_op = [](auto a) { return a; };
    matrix_unary_assignment<TestType>(the_op, corr_op);
}

template<typename TestType>
void tensor3_permute_assignment() {
    auto the_op = [](auto&& out_idx, auto&& rhs_idx, auto&& result, auto&& t0) {
        result.permute_assignment(out_idx, rhs_idx, t0);
    };
    auto corr_op = [](auto a) { return a; };
    tensor3_unary_assignment<TestType>(the_op, corr_op);
}

template<typename TestType>
void tensor4_permute_assignment() {
    auto the_op = [](auto&& out_idx, auto&& rhs_idx, auto&& result, auto&& t0) {
        result.permute_assignment(out_idx, rhs_idx, t0);
    };
    auto corr_op = [](auto a) { return a; };
    tensor4_unary_assignment<TestType>(the_op, corr_op);
}

} // namespace tensorwrapper::testing
