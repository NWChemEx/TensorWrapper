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
#include <tensorwrapper/buffer/contiguous.hpp>
#include <tensorwrapper/generate/generate_eigenvalues.hpp>
#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>
#include <wtf/fp/float_view.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::buffer;
using namespace tensorwrapper::generate;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;

namespace {
double elem_as_double(const Contiguous::const_reference& elem) {
    using wtf::fp::float_cast;
    try {
        return float_cast<double>(elem);
    } catch(const std::runtime_error&) {
        return static_cast<double>(float_cast<float>(elem));
    }
}

std::vector<double> tensor_to_vector(const Tensor& t) {
    auto buf = make_contiguous(t.buffer());
    std::vector<double> rv(buf.shape().extent(0));
    for(std::size_t i = 0; i < rv.size(); ++i) {
        rv[i] = elem_as_double(buf.get_elem({i}));
    }
    return rv;
}
} // namespace

TEST_CASE("generate_eigenvalues") {
    SECTION("linear spacing") {
        SymmetricMatrixSpec spec;
        spec.n                = 4;
        spec.min_eigenvalue   = 1.0;
        spec.condition_number = 10.0;
        spec.spacing          = EigenvalueSpacing::Linear;
        auto gen              = make_rng(1);
        auto result           = generate_eigenvalues(spec, gen);
        auto corr = make_tensor({4}, std::vector<double>{1, 4, 7, 10});
        REQUIRE(approximately_equal(result, corr));
    }

    SECTION("logarithmic spacing") {
        SymmetricMatrixSpec spec;
        spec.n                = 3;
        spec.min_eigenvalue   = 1.0;
        spec.condition_number = 100.0;
        spec.spacing          = EigenvalueSpacing::Logarithmic;
        auto gen              = make_rng(1);
        auto values = tensor_to_vector(generate_eigenvalues(spec, gen));
        REQUIRE(values.size() == 3);
        REQUIRE(values[0] == Catch::Approx(1.0));
        REQUIRE(values[1] == Catch::Approx(10.0));
        REQUIRE(values[2] == Catch::Approx(100.0));
        REQUIRE(std::is_sorted(values.begin(), values.end()));
    }

    SECTION("degenerate spacing") {
        SymmetricMatrixSpec spec;
        spec.n                = 3;
        spec.min_eigenvalue   = 2.5;
        spec.condition_number = 10.0;
        spec.spacing          = EigenvalueSpacing::Degenerate;
        spec.n_clusters       = 1;
        auto gen              = make_rng(1);
        auto values = tensor_to_vector(generate_eigenvalues(spec, gen));
        REQUIRE(values.size() == 3);
        for(const auto v : values) { REQUIRE(v == Catch::Approx(2.5)); }
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
        auto gen              = make_rng(spec.seed);
        auto values = tensor_to_vector(generate_eigenvalues(spec, gen));
        REQUIRE(values.size() == 6);
        REQUIRE(std::is_sorted(values.begin(), values.end()));

        const double lambda_max = spec.min_eigenvalue * spec.condition_number;
        const double dx         = (lambda_max - spec.min_eigenvalue) / 2.0;
        std::vector<double> centers(3);
        for(std::size_t c = 0; c < 3; ++c) {
            centers[c] = spec.min_eigenvalue + static_cast<double>(c) * dx;
        }
        for(const auto v : values) {
            const auto near_center =
              std::any_of(centers.begin(), centers.end(), [&](double center) {
                  return std::abs(v - center) <= spec.cluster_width;
              });
            REQUIRE(near_center);
        }
    }

    SECTION("invalid n throws") {
        SymmetricMatrixSpec spec;
        spec.n   = 0;
        auto gen = make_rng(1);
        REQUIRE_THROWS_AS(generate_eigenvalues(spec, gen),
                          std::invalid_argument);
    }
}
