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
void scalar_binary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    std::vector<value_type> result_data(1, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data(1, value_type{3});
    std::span<value_type> s0_span(s0_data.data(), s0_data.size());

    std::vector<value_type> s1_data(1, value_type{5});
    std::span<value_type> s1_span(s1_data.data(), s1_data.size());

    TestType result(result_span, shape_type({}));
    TestType s0(s0_span, shape_type({}));
    TestType s1(s1_span, shape_type({}));

    label_type out("");
    label_type lhs("");
    label_type rhs("");
    the_op(out, lhs, rhs, result, s0, s1);
    REQUIRE(result.get_elem({}) == corr_op(s0_data[0], s1_data[0]));
}

template<typename TestType, typename Fxn1, typename Fxn2>
void vector_binary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    std::vector<value_type> result_data(4, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data{value_type{1}, value_type{2}, value_type{3},
                                    value_type{4}};
    std::span<value_type> s0_span(s0_data.data(), s0_data.size());

    std::vector<value_type> s1_data{value_type{5}, value_type{6}, value_type{7},
                                    value_type{8}};
    std::span<value_type> s1_span(s1_data.data(), s1_data.size());

    TestType result(result_span, shape_type({4}));
    TestType s0(s0_span, shape_type({4}));
    TestType s1(s1_span, shape_type({4}));

    label_type out("i");
    label_type lhs("i");
    label_type rhs("i");

    the_op(out, lhs, rhs, result, s0, s1);
    for(std::size_t i = 0; i < 4; ++i) {
        REQUIRE(result.get_elem({i}) == corr_op(s0_data[i], s1_data[i]));
    }
}

template<typename TestType, typename Fxn1, typename Fxn2>
void matrix_binary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    std::vector<value_type> result_data(16, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> s0_data(16);
    std::vector<value_type> s1_data(16);
    for(std::size_t i = 0; i < s0_data.size(); ++i) {
        s0_data[i] = static_cast<value_type>(i);
        s1_data[i] = static_cast<value_type>(i * 2);
    }

    std::span<value_type> s0_span(s0_data.data(), s0_data.size());
    std::span<value_type> s1_span(s1_data.data(), s1_data.size());

    TestType result(result_span, shape_type({4, 4}));
    TestType s0(s0_span, shape_type({4, 4}));
    TestType s1(s1_span, shape_type({4, 4}));

    label_type ij("i,j");
    label_type ji("j,i");

    SECTION("No permutation") {
        the_op(ij, ij, ij, result, s0, s1);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                std::size_t idx = i * 4 + j;
                auto corr       = corr_op(s0_data[idx], s1_data[idx]);
                REQUIRE(result.get_elem({i, j}) == corr);
            }
        }
    }

    SECTION("Permute lhs") {
        the_op(ij, ji, ij, result, s0, s1);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                std::size_t lhs_idx = j * 4 + i;
                std::size_t rhs_idx = i * 4 + j;
                auto corr = corr_op(s0_data[lhs_idx], s1_data[rhs_idx]);
                REQUIRE(result.get_elem({i, j}) == corr);
            }
        }
    }

    SECTION("Permute rhs") {
        the_op(ij, ij, ji, result, s0, s1);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                std::size_t lhs_idx = i * 4 + j;
                std::size_t rhs_idx = j * 4 + i;
                auto corr = corr_op(s0_data[lhs_idx], s1_data[rhs_idx]);
                REQUIRE(result.get_elem({i, j}) == corr);
            }
        }
    }

    SECTION("Permute result") {
        the_op(ji, ij, ij, result, s0, s1);
        for(std::size_t i = 0; i < 4; ++i) {
            for(std::size_t j = 0; j < 4; ++j) {
                std::size_t lhs_idx = i * 4 + j;
                std::size_t rhs_idx = i * 4 + j;
                auto corr = corr_op(s0_data[lhs_idx], s1_data[rhs_idx]);
                REQUIRE(result.get_elem({j, i}) == corr);
            }
        }
    }
}

template<typename TestType, typename Fxn1, typename Fxn2>
void tensor3_binary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    const auto n_elements = 8;
    std::vector<value_type> result_data(n_elements, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> t0_data(n_elements);
    std::vector<value_type> t1_data(n_elements);
    for(std::size_t i = 0; i < n_elements; ++i) {
        t0_data[i] = static_cast<value_type>(i);
        t1_data[i] = static_cast<value_type>(i * 2);
    }

    std::span<value_type> t0_span(t0_data.data(), t0_data.size());
    std::span<value_type> t1_span(t1_data.data(), t1_data.size());

    using rank3_index = std::array<std::size_t, 3>;
    std::vector<rank3_index> tensor3_indices;
    for(std::size_t i = 0; i < 2; ++i) {
        for(std::size_t j = 0; j < 2; ++j) {
            for(std::size_t k = 0; k < 2; ++k)
                tensor3_indices.push_back(rank3_index{i, j, k});
        }
    }

    TestType result(result_span, shape_type({2, 2, 2}));
    TestType t0(t0_span, shape_type({2, 2, 2}));
    TestType t1(t1_span, shape_type({2, 2, 2}));

    label_type ijk("i,j,k");
    label_type jik("j,i,k");

    SECTION("No permutation") {
        the_op(ijk, ijk, ijk, result, t0, t1);
        for(auto [i, j, k] : tensor3_indices) {
            std::size_t lhs_idx = i * 4 + j * 2 + k;
            std::size_t rhs_idx = i * 4 + j * 2 + k;
            auto corr           = corr_op(t0_data[lhs_idx], t1_data[rhs_idx]);
            REQUIRE(result.get_elem({i, j, k}) == corr);
        }
    }

    SECTION("Permute lhs") {
        the_op(ijk, jik, ijk, result, t0, t1);
        for(auto [i, j, k] : tensor3_indices) {
            std::size_t lhs_idx = j * 4 + i * 2 + k;
            std::size_t rhs_idx = i * 4 + j * 2 + k;
            auto corr           = corr_op(t0_data[lhs_idx], t1_data[rhs_idx]);
            REQUIRE(result.get_elem({i, j, k}) == corr);
        }
    }

    SECTION("Permute rhs") {
        the_op(ijk, ijk, jik, result, t0, t1);
        for(auto [i, j, k] : tensor3_indices) {
            std::size_t lhs_idx = i * 4 + j * 2 + k;
            std::size_t rhs_idx = j * 4 + i * 2 + k;
            auto corr           = corr_op(t0_data[lhs_idx], t1_data[rhs_idx]);
            REQUIRE(result.get_elem({i, j, k}) == corr);
        }
    }

    SECTION("Permute result") {
        the_op(jik, ijk, ijk, result, t0, t1);
        for(auto [i, j, k] : tensor3_indices) {
            std::size_t lhs_idx = i * 4 + j * 2 + k;
            std::size_t rhs_idx = i * 4 + j * 2 + k;
            auto corr           = corr_op(t0_data[lhs_idx], t1_data[rhs_idx]);
            REQUIRE(result.get_elem({j, i, k}) == corr);
        }
    }
}

template<typename TestType, typename Fxn1, typename Fxn2>
void tensor4_binary_assignment(Fxn1&& the_op, Fxn2&& corr_op) {
    using value_type = typename TestType::value_type;
    using shape_type = typename TestType::shape_type;
    using label_type = typename TestType::label_type;

    const auto n_elements = 16;
    std::vector<value_type> result_data(n_elements, value_type{0});
    std::span<value_type> result_span(result_data.data(), result_data.size());

    std::vector<value_type> t0_data(n_elements);
    std::vector<value_type> t1_data(n_elements);
    for(std::size_t i = 0; i < n_elements; ++i) {
        t0_data[i] = static_cast<value_type>(i);
        t1_data[i] = static_cast<value_type>(i * 2);
    }

    std::span<value_type> t0_span(t0_data.data(), t0_data.size());
    std::span<value_type> t1_span(t1_data.data(), t1_data.size());

    using rank4_index = std::array<std::size_t, 4>;
    std::vector<rank4_index> tensor4_indices;
    for(std::size_t i = 0; i < 2; ++i) {
        for(std::size_t j = 0; j < 2; ++j) {
            for(std::size_t k = 0; k < 2; ++k) {
                for(std::size_t l = 0; l < 2; ++l) {
                    tensor4_indices.emplace_back(rank4_index{i, j, k, l});
                }
            }
        }
    }

    TestType result(result_span, shape_type({2, 2, 2, 2}));
    TestType t0(t0_span, shape_type({2, 2, 2, 2}));
    TestType t1(t1_span, shape_type({2, 2, 2, 2}));

    label_type ijkl("i,j,k,l");
    label_type jilk("j,i,l,k");

    const auto stride0 = 8;
    const auto stride1 = 4;
    const auto stride2 = 2;

    SECTION("No permutation") {
        the_op(ijkl, ijkl, ijkl, result, t0, t1);
        for(auto [i, j, k, l] : tensor4_indices) {
            std::size_t lhs_idx = i * stride0 + j * stride1 + k * stride2 + l;
            std::size_t rhs_idx = i * stride0 + j * stride1 + k * stride2 + l;
            auto corr           = corr_op(t0_data[lhs_idx], t1_data[rhs_idx]);
            REQUIRE(result.get_elem({i, j, k, l}) == corr);
        }
    }

    SECTION("Permute lhs") {
        the_op(ijkl, jilk, ijkl, result, t0, t1);
        for(auto [i, j, k, l] : tensor4_indices) {
            std::size_t lhs_idx = j * stride0 + i * stride1 + l * stride2 + k;
            std::size_t rhs_idx = i * stride0 + j * stride1 + k * stride2 + l;
            auto corr           = corr_op(t0_data[lhs_idx], t1_data[rhs_idx]);
            REQUIRE(result.get_elem({i, j, k, l}) == corr);
        }
    }

    SECTION("Permute rhs") {
        the_op(ijkl, ijkl, jilk, result, t0, t1);
        for(auto [i, j, k, l] : tensor4_indices) {
            std::size_t lhs_idx = i * stride0 + j * stride1 + k * stride2 + l;
            std::size_t rhs_idx = j * stride0 + i * stride1 + l * stride2 + k;
            auto corr           = corr_op(t0_data[lhs_idx], t1_data[rhs_idx]);
            REQUIRE(result.get_elem({i, j, k, l}) == corr);
        }
    }

    SECTION("Permute result") {
        the_op(jilk, ijkl, ijkl, result, t0, t1);
        for(auto [i, j, k, l] : tensor4_indices) {
            std::size_t lhs_idx = i * stride0 + j * stride1 + k * stride2 + l;
            std::size_t rhs_idx = i * stride0 + j * stride1 + k * stride2 + l;
            auto corr           = corr_op(t0_data[lhs_idx], t1_data[rhs_idx]);
            REQUIRE(result.get_elem({j, i, l, k}) == corr);
        }
    }
}

} // namespace tensorwrapper::testing
