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

#include <algorithm>
#include <cmath>
#include <numeric>
#include <tensorwrapper/generate/generate_eigenvalues.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <vector>

namespace tensorwrapper::generate {
namespace {

/** @brief Generates @p n eigenvalues with uniform spacing in @f$\lambda@f$.
 *
 *  @param[in] n Number of eigenvalues.
 *  @param[in] lambda_min Smallest eigenvalue.
 *  @param[in] lambda_max Largest eigenvalue.
 *
 *  @return A vector of length @p n with values from @p lambda_min to
 *          @p lambda_max.
 */
template<concepts::FloatingPoint T>
auto linear_spacing(std::size_t n, double lambda_min, double lambda_max) {
    const auto dx = (lambda_max - lambda_min) / (n - 1);
    std::vector<T> values(n);
    for(std::size_t i = 0; i < n; ++i) {
        values[i] = static_cast<T>(lambda_min + static_cast<double>(i) * dx);
    }
    return values;
}

/** @brief Generates @p n eigenvalues with uniform spacing in @f$\log\lambda@f$.
 *
 *  @param[in] n Number of eigenvalues.
 *  @param[in] lambda_min Smallest eigenvalue. Must be positive.
 *  @param[in] lambda_max Largest eigenvalue. Must be positive.
 *
 *  @return A vector of length @p n with values from @p lambda_min to
 *          @p lambda_max on a log scale.
 */
template<concepts::FloatingPoint T>
auto logarithmic_spacing(std::size_t n, double lambda_min, double lambda_max) {
    const T log_min = types::log(static_cast<T>(lambda_min));
    const T log_max = types::log(static_cast<T>(lambda_max));
    const auto dlog = (log_max - log_min) / static_cast<T>(n - 1);
    std::vector<T> values(n);
    for(std::size_t i = 0; i < n; ++i) {
        values[i] = types::exp(log_min + static_cast<T>(i) * dlog);
    }
    return values;
}

/** @brief Generates @p n eigenvalues grouped into @p n_clusters jittered
 * clusters.
 *
 *  @param[in] n Number of eigenvalues.
 *  @param[in] n_clusters Number of cluster centers.
 *  @param[in] lambda_min Smallest cluster center.
 *  @param[in] lambda_max Largest cluster center.
 *  @param[in] cluster_width Half-width of the uniform jitter around each
 * center.
 *  @param[in,out] gen Random number generator used for the jitter draws.
 *
 *  @return A vector of length @p n with eigenvalues assigned cyclically to
 *          clusters.
 */
template<concepts::FloatingPoint T>
auto clustered_spacing(std::size_t n, std::size_t n_clusters, double lambda_min,
                       double lambda_max, double cluster_width,
                       std::mt19937& gen) {
    std::uniform_real_distribution<double> jitter(-cluster_width,
                                                  cluster_width);
    std::vector<T> values(n);
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
        values[i] = static_cast<T>(cluster_centers[cluster_id] + jitter(gen));
    }
    return values;
}

/** @brief Generates @p n eigenvalues with exact degeneracies within each
 * cluster.
 *
 *  @param[in] n Number of eigenvalues.
 *  @param[in] n_clusters Number of distinct eigenvalue plateaus.
 *  @param[in] lambda_min Value of the smallest plateau.
 *  @param[in] lambda_max Value of the largest plateau.
 *
 *  @return A vector of length @p n with eigenvalues assigned cyclically to
 *          @p n_clusters plateaus.
 */
template<concepts::FloatingPoint T>
auto degenerate_spacing(std::size_t n, std::size_t n_clusters,
                        double lambda_min, double lambda_max) {
    std::vector<T> values(n);
    if(n_clusters <= 1) {
        std::fill(values.begin(), values.end(), static_cast<T>(lambda_min));
    } else {
        const auto n_plateaus = std::min(n_clusters, n);
        const auto nm1        = std::max<std::size_t>(1, n_plateaus - 1);
        const double dx = (lambda_max - lambda_min) / static_cast<double>(nm1);
        for(std::size_t i = 0; i < n; ++i) {
            const auto plateau = i % n_plateaus;
            values[i] =
              static_cast<T>(lambda_min + static_cast<double>(plateau) * dx);
        }
    }
    return values;
}

template<concepts::FloatingPoint T>
double eigenvalue_sort_key(const T& value) {
    if constexpr(types::is_interval_v<T>) {
        return static_cast<double>(value.median());
    } else if constexpr(types::is_uncertain_v<T>) {
        return static_cast<double>(value.mean());
    } else {
        return static_cast<double>(value);
    }
}

template<concepts::FloatingPoint T>
void sort_eigenvalues(std::vector<T>& values) {
    if constexpr(types::is_interval_v<T> || types::is_uncertain_v<T>) {
        std::vector<std::size_t> indices(values.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&](std::size_t a, std::size_t b) {
                      return eigenvalue_sort_key(values[a]) <
                             eigenvalue_sort_key(values[b]);
                  });
        std::vector<T> sorted(values.size());
        for(std::size_t i = 0; i < values.size(); ++i) {
            sorted[i] = values[indices[i]];
        }
        values = std::move(sorted);
    } else {
        std::sort(values.begin(), values.end());
    }
}

} // namespace

template<concepts::FloatingPoint T>
Tensor generate_eigenvalues(const SymmetricMatrixSpec& spec,
                            std::mt19937& gen) {
    require_valid_n(spec.n);
    const auto n = spec.n;

    const double lambda_min = spec.min_eigenvalue;
    const double lambda_max = spec.min_eigenvalue * spec.condition_number;

    if(n == 1) {
        return utilities::make_tensor(
          {n}, std::vector<T>{static_cast<T>(lambda_min)});
    }

    std::vector<T> values;

    switch(spec.spacing) {
        case EigenvalueSpacing::Linear: {
            values = linear_spacing<T>(n, lambda_min, lambda_max);
            break;
        }
        case EigenvalueSpacing::Logarithmic: {
            values = logarithmic_spacing<T>(n, lambda_min, lambda_max);
            break;
        }
        case EigenvalueSpacing::Clustered: {
            const auto n_clusters = std::max<std::size_t>(1, spec.n_clusters);
            const auto n_clusters_clamped = std::min(n_clusters, n);
            values = clustered_spacing<T>(n, n_clusters_clamped, lambda_min,
                                          lambda_max, spec.cluster_width, gen);
            break;
        }
        case EigenvalueSpacing::Degenerate: {
            values =
              degenerate_spacing<T>(n, spec.n_clusters, lambda_min, lambda_max);
            break;
        }
        default: {
            throw std::invalid_argument("Invalid eigenvalue spacing");
        }
    }

    sort_eigenvalues(values);
    return utilities::make_tensor({n}, values);
}

Tensor generate_eigenvalues(const SymmetricMatrixSpec& spec,
                            std::mt19937& gen) {
    return generate_eigenvalues<double>(spec, gen);
}

#define DEFINE_GENERATE_EIGENVALUES(TYPE)       \
    template Tensor generate_eigenvalues<TYPE>( \
      const SymmetricMatrixSpec& spec, std::mt19937& gen);

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_GENERATE_EIGENVALUES);

#undef DEFINE_GENERATE_EIGENVALUES

} // namespace tensorwrapper::generate
