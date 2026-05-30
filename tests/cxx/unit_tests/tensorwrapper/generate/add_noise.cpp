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
} // namespace

TEST_CASE("add_noise") {
    const auto matrix = make_tensor({2, 2}, std::vector<double>{1, 2, 3, 4});

    SECTION("t equals zero returns copy") {
        auto gen = make_rng(1);
        auto out = add_noise(matrix, 0.0, gen);
        REQUIRE(approximately_equal(out, matrix));
    }

    SECTION("within t of input") {
        const double t = 0.05;
        auto gen       = make_rng(17);
        auto out       = add_noise(matrix, t, gen);

        auto in_buf  = make_contiguous(matrix.buffer());
        auto out_buf = make_contiguous(out.buffer());
        for(std::size_t i = 0; i < 2; ++i) {
            for(std::size_t j = 0; j < 2; ++j) {
                const auto x = elem_as_double(in_buf.get_elem({i, j}));
                const auto y = elem_as_double(out_buf.get_elem({i, j}));
                REQUIRE(std::abs(y - x) <= t);
            }
        }
    }

    SECTION("deterministic for fixed seed") {
        const double t = 0.01;
        auto out1      = add_noise(matrix, t, 99);
        auto out2      = add_noise(matrix, t, 99);
        REQUIRE(approximately_equal(out1, out2));
    }

    SECTION("shape preserved") {
        auto gen       = make_rng(3);
        auto out       = add_noise(matrix, 0.01, gen);
        auto in_shape  = make_contiguous(matrix.buffer()).shape();
        auto out_shape = make_contiguous(out.buffer()).shape();
        REQUIRE(out_shape.rank() == in_shape.rank());
        REQUIRE(out_shape.extent(0) == in_shape.extent(0));
        REQUIRE(out_shape.extent(1) == in_shape.extent(1));
    }

    SECTION("invalid t throws") {
        auto gen = make_rng(1);
        REQUIRE_THROWS_AS(add_noise(matrix, -1.0, gen), std::invalid_argument);
    }
}
