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

#include <tensorwrapper/generate/generate_eigen_system.hpp>
#include <tensorwrapper/generate/generate_eigenvalues.hpp>
#include <tensorwrapper/generate/random_orthogonal_matrix.hpp>
#include <tensorwrapper/utilities/diagonal_matrix.hpp>

namespace tensorwrapper::generate {

EigenSystem generate_eigen_system(const SymmetricMatrixSpec& spec) {
    require_valid_n(spec.n);
    auto gen     = make_rng(spec.seed);
    const auto n = spec.n;

    EigenSystem rv;
    rv.n            = n;
    rv.eigenvalues  = generate_eigenvalues(spec, gen);
    rv.eigenvectors = random_orthogonal_matrix(n, gen);

    const auto D = utilities::diagonal_matrix(rv.eigenvalues);

    Tensor qd;
    qd("i,k") = rv.eigenvectors("i,l") * D("l,k");

    Tensor matrix;
    matrix("i,j") = qd("i,k") * rv.eigenvectors("j,k");
    rv.matrix     = std::move(matrix);
    return rv;
}

} // namespace tensorwrapper::generate
