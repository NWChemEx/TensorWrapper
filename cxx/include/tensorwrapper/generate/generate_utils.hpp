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

/// Maximum supported dimension for generated square matrices.
constexpr std::size_t kMaxMatrixDim = 10;

/** @brief Specifies how eigenvalues are distributed between the endpoints.
 *
 *  Each spacing mode fills the interval
 *  @f$[\lambda_{\min}, \lambda_{\max}]@f$ with @f$n@f$ eigenvalues, where
 *  @f$\lambda_{\max} = \lambda_{\min} \times@f$ the condition number.
 */
enum class EigenvalueSpacing {
    /// Uniform spacing with @f$\Delta\lambda = (\lambda_{\max} -
    /// \lambda_{\min}) / (n - 1)@f$.
    Linear,
    /// Uniform spacing in log space with
    /// @f$\Delta\log\lambda = \log(\lambda_{\max} / \lambda_{\min}) / (n -
    /// 1)@f$.
    Logarithmic,
    /// Eigenvalues are grouped into clusters of width @p cluster_width that are
    /// separated by @f$(\lambda_{\max} - \lambda_{\min}) / (n_{\text{clusters}}
    /// - 1)@f$.
    Clustered,
    /// Same cluster centers as @ref Clustered, but all eigenvalues in a cluster
    /// are identical.
    Degenerate
};

/** @brief Parameters controlling the generation of a symmetric test matrix.
 *
 *  The resulting matrix has eigenvalues in
 *  @f$[\texttt{min\_eigenvalue},
 *  \texttt{min\_eigenvalue} \times \texttt{condition\_number}]@f$ with the
 *  spacing determined by @p spacing.
 */
struct SymmetricMatrixSpec {
    /// Dimension of the square matrix.
    std::size_t n = 2;
    /// Ratio of the largest to smallest eigenvalue.
    double condition_number = 10.0;
    /// Smallest eigenvalue in the spectrum.
    double min_eigenvalue = 1.0;
    /// Strategy used to distribute eigenvalues between the endpoints.
    EigenvalueSpacing spacing = EigenvalueSpacing::Linear;
    /// Number of eigenvalue clusters when @p spacing is @ref Clustered or
    /// @ref Degenerate.
    std::size_t n_clusters = 1;
    /// Half-width of each cluster when @p spacing is @ref Clustered.
    double cluster_width = 1e-6;
    /// Seed for random number generation. A value of zero selects a
    /// non-deterministic seed.
    std::uint64_t seed = 42;
};

/** @brief Creates a Mersenne Twister RNG from @p seed.
 *
 *  @param[in] seed The seed value. When zero, a seed is drawn from
 *                  `std::random_device`.
 *
 *  @return A `std::mt19937` generator initialized with @p seed.
 */
inline std::mt19937 make_rng(std::uint64_t seed) {
    if(seed == 0) {
        std::random_device rd;
        return std::mt19937(rd());
    }
    return std::mt19937(static_cast<std::mt19937::result_type>(seed));
}

/** @brief Validates that @p n is an allowed matrix dimension.
 *
 *  @param[in] n The dimension to validate.
 *
 *  @throw std::invalid_argument if @p n is not in `[1, kMaxMatrixDim]`.
 */
inline void require_valid_n(std::size_t n) {
    if(n < 1 || n > kMaxMatrixDim) {
        throw std::invalid_argument("n must be in [1, kMaxMatrixDim]");
    }
}

} // namespace tensorwrapper::generate
