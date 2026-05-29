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
#include <algorithm>
#include <cmath>
#include <tensorwrapper/generate/generate_utils.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <vector>

namespace tensorwrapper::generate {
namespace {
inline auto linear_spacing(std::size_t n, double lambda_min,
                           double lambda_max) {
    const auto dx = (lambda_max - lambda_min) / (n - 1);
    std::vector<double> values(n);
    for(std::size_t i = 0; i < n; ++i) {
        values[i] = lambda_min + static_cast<double>(i) * dx;
    }
    return values;
}

inline auto logarithmic_spacing(std::size_t n, double lambda_min,
                                double lambda_max) {
    const double log_min = std::log(lambda_min);
    const double log_max = std::log(lambda_max);
    const double dlog    = (log_max - log_min) / (n - 1);
    std::vector<double> values(n);
    for(std::size_t i = 0; i < n; ++i) {
        values[i] = std::exp(log_min + static_cast<double>(i) * dlog);
    }
    return values;
}

inline auto clustered_spacing(std::size_t n, std::size_t n_clusters,
                              double lambda_min, double lambda_max,
                              double cluster_width, std::mt19937& gen) {
    std::uniform_real_distribution<double> jitter(-cluster_width,
                                                  cluster_width);
    std::vector<double> values(n);
    std::vector<double> cluster_centers(n_clusters);
    if(n_clusters == 1) {
        cluster_centers[0] = lambda_min;
    } else {
        const double dx = (lambda_max - lambda_min) / (n_clusters - 1);
        for(std::size_t c = 0; c < n_clusters; ++c) {
            cluster_centers[c] = lambda_min + static_cast<double>(c) * dx;
        }
    }

    for(std::size_t i = 0; i < n; ++i) {
        const auto cluster_id = i % n_clusters;
        values[i]             = cluster_centers[cluster_id] + jitter(gen);
    }
    return values;
}

inline auto degenerate_spacing(std::size_t n, std::size_t n_clusters,
                               double lambda_min, double lambda_max) {
    std::vector<double> values(n);
    if(n_clusters <= 1) {
        std::fill(values.begin(), values.end(), lambda_min);
    } else {
        const auto n_plateaus = std::min(n_clusters, n);
        const auto nm1        = std::max<std::size_t>(1, n_plateaus - 1);
        const double dx = (lambda_max - lambda_min) / static_cast<double>(nm1);
        for(std::size_t i = 0; i < n; ++i) {
            const auto plateau = i % n_plateaus;
            values[i]          = lambda_min + static_cast<double>(plateau) * dx;
        }
    }
    return values;
}
} // namespace

inline Tensor generate_eigenvalues(const SymmetricMatrixSpec& spec,
                                   std::mt19937& gen) {
    require_valid_n(spec.n);
    const auto n = spec.n;

    const double lambda_min = spec.min_eigenvalue;
    const double lambda_max = spec.min_eigenvalue * spec.condition_number;

    std::vector<double> values;

    if(n == 1) return utilities::make_tensor({n}, values);

    switch(spec.spacing) {
        case EigenvalueSpacing::Linear: {
            values = linear_spacing(n, lambda_min, lambda_max);
            break;
        }
        case EigenvalueSpacing::Logarithmic: {
            values = logarithmic_spacing(n, lambda_min, lambda_max);
            break;
        }
        case EigenvalueSpacing::Clustered: {
            const auto n_clusters = std::max<std::size_t>(1, spec.n_clusters);
            const auto n_clusters_clamped = std::min(n_clusters, n);
            values = clustered_spacing(n, n_clusters_clamped, lambda_min,
                                       lambda_max, spec.cluster_width, gen);
            break;
        }
        case EigenvalueSpacing::Degenerate: {
            values =
              degenerate_spacing(n, spec.n_clusters, lambda_min, lambda_max);
            break;
        }
        default: {
            throw std::invalid_argument("Invalid eigenvalue spacing");
        }
    }

    std::sort(values.begin(), values.end());
    return utilities::make_tensor({n}, values);
}

} // namespace tensorwrapper::generate
