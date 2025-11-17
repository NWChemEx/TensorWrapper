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
#include <span>
#include <vector>

namespace tensorwrapper::testing {

template<typename TestType, typename Fxn1, typename Fxn2>
void scalar_unary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    std::vector<value_type> result_data(1, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data(1, value_type{3});
    std::span<value_type> s0_span(s0_data.data(), s0_data.size());

    TestType result(result_span, shape_type({}));
    TestType s0(s0_span, shape_type({}));

    label_type out("");
    label_type rhs("");
    the_op(out, rhs, result, s0);
    REQUIRE(result.get_elem({}) == corr_op(s0_data[0]));
}

template<typename TestType, typename Fxn1, typename Fxn2>
void vector_unary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    const auto n_elements = 4;
    std::vector<value_type> result_data(n_elements, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data(n_elements, value_type{3});
    std::span<value_type> s0_span(s0_data.data(), s0_data.size());

    TestType result(result_span, shape_type({4}));
    TestType s0(s0_span, shape_type({4}));

    label_type out("i");
    label_type rhs("i");
    the_op(out, rhs, result, s0);
    for(std::size_t i = 0; i < n_elements; ++i)
        REQUIRE(result.get_elem({i}) == corr_op(s0_data[i]));
}

template<typename TestType, typename Fxn1, typename Fxn2>
void matrix_unary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    const auto n_elements = 16;
    std::vector<value_type> result_data(n_elements, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data(n_elements, value_type{0});
    for(std::size_t i = 0; i < n_elements; ++i)
        s0_data[i] = static_cast<value_type>(i);

    std::span<value_type> s0_span(s0_data.data(), s0_data.size());

    TestType result(result_span, shape_type({4, 4}));
    TestType s0(s0_span, shape_type({4, 4}));

    label_type ij("i,j");
    label_type ji("j,i");

    SECTION("No permutation") {
        the_op(ij, ij, result, s0);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < 4; ++j) {
                std::size_t idx = i * 4 + j;
                REQUIRE(result.get_elem({i, j}) == corr_op(s0_data[idx]));
            }
    }

    SECTION("Permute rhs") {
        the_op(ij, ji, result, s0);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < 4; ++j) {
                std::size_t idx = j * 4 + i;
                REQUIRE(result.get_elem({i, j}) == corr_op(s0_data[idx]));
            }
    }

    SECTION("Permute result") {
        the_op(ji, ij, result, s0);
        for(std::size_t i = 0; i < 4; ++i)
            for(std::size_t j = 0; j < 4; ++j) {
                std::size_t idx = i * 4 + j;
                REQUIRE(result.get_elem({j, i}) == corr_op(s0_data[idx]));
            }
    }
}

template<typename TestType, typename Fxn1, typename Fxn2>
void tensor3_unary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    const auto n_elements = 8;
    std::vector<value_type> result_data(n_elements, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data(n_elements, value_type{0});
    for(std::size_t i = 0; i < n_elements; ++i)
        s0_data[i] = static_cast<value_type>(i);

    std::span<value_type> s0_span(s0_data.data(), s0_data.size());

    TestType result(result_span, shape_type({2, 2, 2}));
    TestType s0(s0_span, shape_type({2, 2, 2}));

    label_type ijk("i,j,k");
    label_type jik("j,i,k");

    using rank3_index = std::array<std::size_t, 3>;
    std::vector<rank3_index> tensor3_indices;
    for(std::size_t i = 0; i < 2; ++i) {
        for(std::size_t j = 0; j < 2; ++j) {
            for(std::size_t k = 0; k < 2; ++k)
                tensor3_indices.push_back(rank3_index{i, j, k});
        }
    }

    SECTION("No permutation") {
        the_op(ijk, ijk, result, s0);
        for(const auto [i, j, k] : tensor3_indices) {
            std::size_t idx = i * 4 + j * 2 + k;
            REQUIRE(result.get_elem({i, j, k}) == corr_op(s0_data[idx]));
        }
    }

    SECTION("Permute rhs") {
        the_op(ijk, jik, result, s0);
        for(const auto [i, j, k] : tensor3_indices) {
            std::size_t idx = j * 4 + i * 2 + k;
            REQUIRE(result.get_elem({i, j, k}) == corr_op(s0_data[idx]));
        }
    }

    SECTION("Permute result") {
        the_op(jik, ijk, result, s0);
        for(const auto [i, j, k] : tensor3_indices) {
            std::size_t idx = i * 4 + j * 2 + k;
            REQUIRE(result.get_elem({j, i, k}) == corr_op(s0_data[idx]));
        }
    }
}

template<typename TestType, typename Fxn1, typename Fxn2>
void tensor4_unary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    const auto n_elements = 16;
    std::vector<value_type> result_data(n_elements, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data(n_elements, value_type{0});
    for(std::size_t i = 0; i < n_elements; ++i)
        s0_data[i] = static_cast<value_type>(i);

    std::span<value_type> s0_span(s0_data.data(), s0_data.size());

    TestType result(result_span, shape_type({2, 2, 2, 2}));
    TestType s0(s0_span, shape_type({2, 2, 2, 2}));

    label_type ijkl("i,j,k,l");
    label_type jikl("j,i,k,l");

    using rank4_index = std::array<std::size_t, 4>;
    std::vector<rank4_index> tensor4_indices;
    for(std::size_t i = 0; i < 2; ++i) {
        for(std::size_t j = 0; j < 2; ++j) {
            for(std::size_t k = 0; k < 2; ++k)
                for(std::size_t l = 0; l < 2; ++l)
                    tensor4_indices.push_back(rank4_index{i, j, k, l});
        }
    }

    SECTION("No permutation") {
        the_op(ijkl, ijkl, result, s0);
        for(const auto [i, j, k, l] : tensor4_indices) {
            std::size_t idx = i * 8 + j * 4 + k * 2 + l;
            REQUIRE(result.get_elem({i, j, k, l}) == corr_op(s0_data[idx]));
        }
    }

    SECTION("Permute rhs") {
        the_op(ijkl, jikl, result, s0);
        for(const auto [i, j, k, l] : tensor4_indices) {
            std::size_t idx = j * 8 + i * 4 + k * 2 + l;
            REQUIRE(result.get_elem({i, j, k, l}) == corr_op(s0_data[idx]));
        }
    }

    SECTION("Permute result") {
        the_op(jikl, ijkl, result, s0);
        for(const auto [i, j, k, l] : tensor4_indices) {
            std::size_t idx = i * 8 + j * 4 + k * 2 + l;
            REQUIRE(result.get_elem({j, i, k, l}) == corr_op(s0_data[idx]));
        }
    }
}

} // namespace tensorwrapper::testing
