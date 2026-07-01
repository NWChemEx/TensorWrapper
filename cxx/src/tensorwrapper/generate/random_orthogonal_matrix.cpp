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

#include <Eigen/Dense>
#include <tensorwrapper/generate/generate_utils.hpp>
#include <tensorwrapper/generate/random_orthogonal_matrix.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <vector>

namespace tensorwrapper::generate {
namespace {

Eigen::MatrixXd random_orthogonal_matrix_eigen(std::size_t n,
                                               std::mt19937& gen) {
    Eigen::MatrixXd M(n, n);
    std::normal_distribution<double> dist(0.0, 1.0);
    for(std::size_t i = 0; i < n; ++i) {
        for(std::size_t j = 0; j < n; ++j) {
            M(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)) =
              dist(gen);
        }
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(M);
    return Eigen::MatrixXd(qr.householderQ());
}

} // namespace

template<concepts::FloatingPoint T>
Tensor random_orthogonal_matrix(std::size_t n, std::mt19937& gen) {
    require_valid_n(n);

    const auto Q = random_orthogonal_matrix_eigen(n, gen);

    std::vector<T> data(n * n);
    for(std::size_t i = 0; i < n; ++i) {
        for(std::size_t j = 0; j < n; ++j) {
            data[i * n + j] = static_cast<T>(
              Q(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(j)));
        }
    }
    return utilities::make_tensor({n, n}, data);
}

Tensor random_orthogonal_matrix(std::size_t n, std::mt19937& gen) {
    return random_orthogonal_matrix<double>(n, gen);
}

#define DEFINE_RANDOM_ORTHOGONAL_MATRIX(TYPE)                     \
    template Tensor random_orthogonal_matrix<TYPE>(std::size_t n, \
                                                   std::mt19937& gen);

TW_APPLY_FLOATING_POINT_TYPES(DEFINE_RANDOM_ORTHOGONAL_MATRIX);

#undef DEFINE_RANDOM_ORTHOGONAL_MATRIX

} // namespace tensorwrapper::generate
