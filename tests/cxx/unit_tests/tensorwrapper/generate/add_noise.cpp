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
#include <tensorwrapper/generate/add_noise.hpp>
#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/types/floating_point.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>
#include <wtf/fp/float_view.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::buffer;
using namespace tensorwrapper::generate;
using namespace tensorwrapper::operations;
using namespace tensorwrapper::utilities;

namespace {

template<typename T>
void require_within_noise(const T& in, const T& out, double t) {
    using tensorwrapper::types::uq_center;
    REQUIRE(std::abs(uq_center(in) - uq_center(out)) <= t);
}
} // namespace

TEMPLATE_LIST_TEST_CASE("add_noise", "", types::floating_point_types) {
    const auto matrix =
      make_tensor({2, 2}, std::vector<TestType>{TestType{1}, TestType{2},
                                                TestType{3}, TestType{4}});

    SECTION("t equals zero returns copy") {
        auto gen = make_rng(1);
        auto out = add_noise<TestType>(matrix, 0.0, gen);
        REQUIRE(approximately_equal(out, matrix));
    }

    SECTION("within t of input") {
        const double t = 0.05;
        auto gen       = make_rng(17);
        auto out       = add_noise<TestType>(matrix, t, gen);

        using wtf::fp::float_cast;
        auto in_buf  = make_contiguous(matrix.buffer());
        auto out_buf = make_contiguous(out.buffer());
        for(std::size_t i = 0; i < 2; ++i) {
            for(std::size_t j = 0; j < 2; ++j) {
                const auto in_val =
                  float_cast<TestType>(in_buf.get_elem({i, j}));
                const auto out_val =
                  float_cast<TestType>(out_buf.get_elem({i, j}));
                require_within_noise(in_val, out_val, t);
            }
        }
    }

    SECTION("deterministic for fixed seed") {
        const double t = 0.01;
        auto out1      = add_noise<TestType>(matrix, t, 99);
        auto out2      = add_noise<TestType>(matrix, t, 99);
        if constexpr(types::is_interval_v<TestType>) {
            using wtf::fp::float_cast;
            auto b1 = make_contiguous(out1.buffer());
            auto b2 = make_contiguous(out2.buffer());
            for(std::size_t i = 0; i < 2; ++i) {
                for(std::size_t j = 0; j < 2; ++j) {
                    const auto v1 = float_cast<TestType>(b1.get_elem({i, j}));
                    const auto v2 = float_cast<TestType>(b2.get_elem({i, j}));
                    REQUIRE(v1.lower() == Catch::Approx(v2.lower()));
                    REQUIRE(v1.upper() == Catch::Approx(v2.upper()));
                }
            }
        } else {
            REQUIRE(approximately_equal(out1, out2));
        }
    }

    SECTION("shape preserved") {
        auto gen       = make_rng(3);
        auto out       = add_noise<TestType>(matrix, 0.01, gen);
        auto in_shape  = make_contiguous(matrix.buffer()).shape();
        auto out_shape = make_contiguous(out.buffer()).shape();
        REQUIRE(out_shape.rank() == in_shape.rank());
        REQUIRE(out_shape.extent(0) == in_shape.extent(0));
        REQUIRE(out_shape.extent(1) == in_shape.extent(1));
    }

    SECTION("invalid t throws") {
        auto gen = make_rng(1);
        REQUIRE_THROWS_AS(add_noise<TestType>(matrix, -1.0, gen),
                          std::invalid_argument);
    }
}
