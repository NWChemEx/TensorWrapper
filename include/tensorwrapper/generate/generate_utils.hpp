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

#pragma once
#include <cstdint>
#include <random>
#include <stdexcept>

namespace tensorwrapper::generate {

constexpr std::size_t kMaxMatrixDim = 10;

enum class EigenvalueSpacing {
    // delta = (lambda_max - lambda_min) / (n - 1).
    Linear,
    // delta = log(lambda_max / lambda_min) / (n - 1).
    Logarithmic,
    // clusters of width cluster_width are separated by
    // (lambda_max - lambda_min) / (n_clusters - 1).
    Clustered,
    // same as clustered, but all eigenvalues in a cluster are the same.
    Degenerate
};

struct SymmetricMatrixSpec {
    std::size_t n             = 2;
    double condition_number   = 10.0;
    double min_eigenvalue     = 1.0;
    EigenvalueSpacing spacing = EigenvalueSpacing::Linear;
    std::size_t n_clusters    = 1;
    double cluster_width      = 1e-6;
    std::uint64_t seed        = 42;
};

inline std::mt19937 make_rng(std::uint64_t seed) {
    if(seed == 0) {
        std::random_device rd;
        return std::mt19937(rd());
    }
    return std::mt19937(static_cast<std::mt19937::result_type>(seed));
}

inline void require_valid_n(std::size_t n) {
    if(n < 1 || n > kMaxMatrixDim) {
        throw std::invalid_argument("n must be in [1, kMaxMatrixDim]");
    }
}

} // namespace tensorwrapper::generate
