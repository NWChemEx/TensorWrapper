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

#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/generate/generate_eigen_system.hpp>
#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <tensorwrapper/utilities/diagonal_matrix.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::buffer;
using namespace tensorwrapper::generate;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;

namespace {
template<typename T>
constexpr double eigen_system_tol =
  std::is_same_v<T, float> || std::is_same_v<T, types::ufloat> ||
      std::is_same_v<T, types::ifloat> ?
    1e-5 :
    1e-12;
} // namespace

TEMPLATE_LIST_TEST_CASE("generate_eigen_system", "",
                        types::floating_point_types) {
    SECTION("shapes") {
        SymmetricMatrixSpec spec;
        spec.n                = 4;
        spec.condition_number = 1e3;
        spec.spacing          = EigenvalueSpacing::Linear;
        spec.seed             = 11;
        auto system           = generate_eigen_system<TestType>(spec);

        REQUIRE(system.n == 4);
        REQUIRE(
          make_contiguous(system.eigenvalues.buffer()).shape().extent(0) == 4);
        REQUIRE(
          make_contiguous(system.eigenvectors.buffer()).shape().extent(0) == 4);
        REQUIRE(
          make_contiguous(system.eigenvectors.buffer()).shape().extent(1) == 4);
        REQUIRE(make_contiguous(system.matrix.buffer()).shape().extent(0) == 4);
        REQUIRE(make_contiguous(system.matrix.buffer()).shape().extent(1) == 4);
    }

    SECTION("symmetry") {
        SymmetricMatrixSpec spec;
        spec.n      = 3;
        spec.seed   = 5;
        auto system = generate_eigen_system<TestType>(spec);

        Tensor sym;
        sym("i,j") = system.matrix("i,j");
        Tensor tran;
        tran("i,j") = system.matrix("j,i");
        REQUIRE(approximately_equal(sym, tran, eigen_system_tol<TestType>));
    }

    SECTION("orthonormal eigenvectors") {
        SymmetricMatrixSpec spec;
        spec.n      = 3;
        spec.seed   = 7;
        auto system = generate_eigen_system<TestType>(spec);

        Tensor product;
        product("i,k") =
          system.eigenvectors("i,j") * system.eigenvectors("k,j");

        auto ones = make_tensor(
          {3}, std::vector<TestType>{TestType{1}, TestType{1}, TestType{1}});
        auto ident = diagonal_matrix(ones);
        REQUIRE(
          approximately_equal(product, ident, eigen_system_tol<TestType>));
    }

    SECTION("exact eigenpair") {
        SymmetricMatrixSpec spec;
        spec.n      = 4;
        spec.seed   = 13;
        auto system = generate_eigen_system<TestType>(spec);

        Tensor av;
        av("i,k") = system.matrix("i,j") * system.eigenvectors("j,k");

        const auto D = diagonal_matrix(system.eigenvalues);
        Tensor scaled;
        scaled("i,k") = system.eigenvectors("i,l") * D("l,k");
        REQUIRE(approximately_equal(av, scaled, eigen_system_tol<TestType>));
    }

    SECTION("deterministic for fixed seed") {
        SymmetricMatrixSpec spec;
        spec.n       = 4;
        spec.seed    = 17;
        auto system1 = generate_eigen_system<TestType>(spec);
        auto system2 = generate_eigen_system<TestType>(spec);
        REQUIRE(approximately_equal(system1.eigenvalues, system2.eigenvalues));
        REQUIRE(
          approximately_equal(system1.eigenvectors, system2.eigenvectors));
        REQUIRE(approximately_equal(system1.matrix, system2.matrix,
                                    eigen_system_tol<TestType>));
    }

    SECTION("degenerate") {
        SymmetricMatrixSpec spec;
        spec.n                = 2;
        spec.condition_number = 1.0;
        spec.min_eigenvalue   = 1.0;
        spec.spacing          = EigenvalueSpacing::Degenerate;
        spec.n_clusters       = 1;
        spec.seed             = 7;
        auto system           = generate_eigen_system<TestType>(spec);

        auto evals =
          make_tensor({2}, std::vector<TestType>{TestType{1.0}, TestType{1.0}});
        REQUIRE(approximately_equal(system.eigenvalues, evals));

        auto ident =
          make_tensor({2, 2}, std::vector<TestType>{TestType{1}, TestType{0},
                                                    TestType{0}, TestType{1}});
        REQUIRE(approximately_equal(system.matrix, ident,
                                    eigen_system_tol<TestType>));
    }

    SECTION("invalid n throws") {
        SymmetricMatrixSpec spec;
        spec.n = 0;
        REQUIRE_THROWS_AS(generate_eigen_system<TestType>(spec),
                          std::invalid_argument);
    }
}
