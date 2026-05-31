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

#include <tensorwrapper/generate/generate_eigenvalues.hpp>
#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::generate;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;

TEMPLATE_LIST_TEST_CASE("generate_eigenvalues", "",
                        types::floating_point_types) {
    SECTION("n == 1") {
        SymmetricMatrixSpec spec;
        spec.n              = 1;
        spec.min_eigenvalue = 3.5;
        auto gen            = make_rng(1);
        auto result         = generate_eigenvalues<TestType>(spec, gen);
        auto corr = make_tensor({1}, std::vector<TestType>{TestType{3.5}});
        REQUIRE(approximately_equal(result, corr));
    }

    SECTION("linear spacing") {
        SymmetricMatrixSpec spec;
        spec.n                = 4;
        spec.min_eigenvalue   = 1.0;
        spec.condition_number = 10.0;
        spec.spacing          = EigenvalueSpacing::Linear;
        auto gen              = make_rng(1);
        auto result           = generate_eigenvalues<TestType>(spec, gen);
        auto corr =
          make_tensor({4}, std::vector<TestType>{TestType{1}, TestType{4},
                                                 TestType{7}, TestType{10}});
        REQUIRE(approximately_equal(result, corr));
    }

    SECTION("logarithmic spacing") {
        SymmetricMatrixSpec spec;
        spec.n                = 3;
        spec.min_eigenvalue   = 1.0;
        spec.condition_number = 100.0;
        spec.spacing          = EigenvalueSpacing::Logarithmic;
        const auto lambda_min = static_cast<TestType>(spec.min_eigenvalue);
        const auto lambda_max =
          static_cast<TestType>(spec.min_eigenvalue * spec.condition_number);
        const auto log_min = types::log(lambda_min);
        const auto log_max = types::log(lambda_max);
        const auto dlog    = (log_max - log_min) / TestType{2};
        std::vector<TestType> expected(3);
        for(std::size_t i = 0; i < 3; ++i) {
            expected[i] = types::exp(log_min + static_cast<TestType>(i) * dlog);
        }
        auto gen    = make_rng(1);
        auto result = generate_eigenvalues<TestType>(spec, gen);
        auto corr   = make_tensor({3}, expected);
        REQUIRE(approximately_equal(result, corr));
    }

    SECTION("degenerate spacing") {
        SymmetricMatrixSpec spec;
        spec.n                = 3;
        spec.min_eigenvalue   = 2.5;
        spec.condition_number = 10.0;
        spec.spacing          = EigenvalueSpacing::Degenerate;
        spec.n_clusters       = 1;
        auto gen              = make_rng(1);
        auto result           = generate_eigenvalues<TestType>(spec, gen);
        auto corr =
          make_tensor({3}, std::vector<TestType>{TestType{2.5}, TestType{2.5},
                                                 TestType{2.5}});
        REQUIRE(approximately_equal(result, corr));
    }

    SECTION("clustered spacing") {
        SymmetricMatrixSpec spec;
        spec.n                = 6;
        spec.min_eigenvalue   = 1.0;
        spec.condition_number = 100.0;
        spec.spacing          = EigenvalueSpacing::Clustered;
        spec.n_clusters       = 3;
        spec.cluster_width    = 1e-6;
        spec.seed             = 23;

        auto gen    = make_rng(spec.seed);
        auto result = generate_eigenvalues<TestType>(spec, gen);

        gen              = make_rng(spec.seed);
        auto result_copy = generate_eigenvalues<TestType>(spec, gen);
        REQUIRE(approximately_equal(result, result_copy));

        SymmetricMatrixSpec plateau_spec = spec;
        plateau_spec.spacing             = EigenvalueSpacing::Degenerate;
        auto plateau_gen                 = make_rng(1);
        auto plateaus =
          generate_eigenvalues<TestType>(plateau_spec, plateau_gen);
        REQUIRE(approximately_equal(result, plateaus, spec.cluster_width));
    }

    SECTION("invalid n throws") {
        SymmetricMatrixSpec spec;
        spec.n   = 0;
        auto gen = make_rng(1);
        REQUIRE_THROWS_AS(generate_eigenvalues<TestType>(spec, gen),
                          std::invalid_argument);
    }
}
