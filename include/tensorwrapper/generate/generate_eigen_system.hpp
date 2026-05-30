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
#include <cstddef>
#include <tensorwrapper/generate/generate_utils.hpp>
#include <tensorwrapper/tensor/tensor.hpp>

namespace tensorwrapper::generate {

/** @brief A symmetric matrix together with its eigen-decomposition.
 *
 *  The member @p matrix satisfies @f$M = Q D Q^T@f$ where @p eigenvectors is
 *  @f$Q@f$ and @p eigenvalues holds the diagonal entries of @f$D@f$.
 */
struct EigenSystem {
    /// Dimension of the square matrix.
    std::size_t n = 0;
    /// Rank-1 tensor of length @p n containing the eigenvalues.
    Tensor eigenvalues;
    /// Rank-2 tensor of shape `(n, n)` whose columns are the eigenvectors.
    Tensor eigenvectors;
    /// Rank-2 tensor of shape `(n, n)` representing the symmetric matrix.
    Tensor matrix;
};

/** @brief Generates a reproducible symmetric matrix and its
 * eigen-decomposition.
 *
 *  Constructs @f$M = Q D Q^T@f$ where @f$D@f$ is built from eigenvalues
 *  generated according to @p spec and @f$Q@f$ is a random orthogonal matrix.
 *
 *  @param[in] spec Parameters controlling the matrix dimension, spectrum, and
 *                  random seed.
 *
 *  @return An @ref EigenSystem populated with @p spec.n, the eigenvalues,
 *          eigenvectors, and the assembled matrix.
 *
 *  @throw std::invalid_argument if @p spec.n is outside `[1, kMaxMatrixDim]`.
 */
EigenSystem generate_eigen_system(const SymmetricMatrixSpec& spec);

} // namespace tensorwrapper::generate
