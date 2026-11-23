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

#include <tensorwrapper/operations/approximately_equal.hpp>
#include <tensorwrapper/operations/stack.hpp>
#include <tensorwrapper/utilities/make_tensor.hpp>
#include <testing/testing.hpp>

using namespace tensorwrapper;
using namespace tensorwrapper::operations;

namespace {

// Makes a tensor of FloatType with the given shape and (row-major) values
template<typename FloatType>
Tensor make(std::vector<std::size_t> shape, std::vector<double> values) {
    std::vector<FloatType> data;
    for(auto v : values) data.push_back(FloatType(v));
    return tensorwrapper::utilities::make_tensor(shape, data.begin(),
                                                 data.end());
}

// Checks that rv has the same shape as corr and the same elements
template<typename FloatType>
void check(const Tensor& rv, const Tensor& corr) {
    const auto& rv_buffer   = buffer::make_contiguous(rv.buffer());
    const auto& corr_buffer = buffer::make_contiguous(corr.buffer());
    REQUIRE(rv_buffer.shape() == corr_buffer.shape());
    REQUIRE(
      approximately_equal(rv, corr, testing::default_tolerance<FloatType>()));
}

} // namespace

TEMPLATE_LIST_TEST_CASE("stack", "", types::floating_point_types) {
    using error_t = std::runtime_error;

    SECTION("scalars") {
        std::vector<Tensor> ts{make<TestType>({}, {1.0}),
                               make<TestType>({}, {2.0}),
                               make<TestType>({}, {3.0})};
        auto corr = make<TestType>({3}, {1.0, 2.0, 3.0});
        check<TestType>(stack(ts), corr);
        check<TestType>(stack(ts, -1), corr);
        REQUIRE_THROWS_AS(stack(ts, 1), error_t);
        REQUIRE_THROWS_AS(stack(ts, -2), error_t);
    }

    SECTION("vectors") {
        std::vector<Tensor> ts{make<TestType>({3}, {0.0, 1.0, 2.0}),
                               make<TestType>({3}, {3.0, 4.0, 5.0})};

        auto corr0 = make<TestType>({2, 3}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0});
        check<TestType>(stack(ts), corr0);
        check<TestType>(stack(ts, 0), corr0);
        check<TestType>(stack(ts, -2), corr0);

        auto corr1 = make<TestType>({3, 2}, {0.0, 3.0, 1.0, 4.0, 2.0, 5.0});
        check<TestType>(stack(ts, 1), corr1);
        check<TestType>(stack(ts, -1), corr1);

        REQUIRE_THROWS_AS(stack(ts, 2), error_t);
        REQUIRE_THROWS_AS(stack(ts, -3), error_t);
    }

    SECTION("matrices") {
        std::vector<Tensor> ts{
          make<TestType>({2, 3}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0}),
          make<TestType>({2, 3}, {6.0, 7.0, 8.0, 9.0, 10.0, 11.0})};

        auto corr0 =
          make<TestType>({2, 2, 3}, {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                     8.0, 9.0, 10.0, 11.0});
        check<TestType>(stack(ts, 0), corr0);
        check<TestType>(stack(ts, -3), corr0);

        auto corr1 =
          make<TestType>({2, 2, 3}, {0.0, 1.0, 2.0, 6.0, 7.0, 8.0, 3.0, 4.0,
                                     5.0, 9.0, 10.0, 11.0});
        check<TestType>(stack(ts, 1), corr1);
        check<TestType>(stack(ts, -2), corr1);

        auto corr2 =
          make<TestType>({2, 3, 2}, {0.0, 6.0, 1.0, 7.0, 2.0, 8.0, 3.0, 9.0,
                                     4.0, 10.0, 5.0, 11.0});
        check<TestType>(stack(ts, 2), corr2);
        check<TestType>(stack(ts, -1), corr2);

        REQUIRE_THROWS_AS(stack(ts, 3), error_t);
        REQUIRE_THROWS_AS(stack(ts, -4), error_t);
    }

    SECTION("single tensor") {
        std::vector<Tensor> ts{make<TestType>({3}, {0.0, 1.0, 2.0})};
        check<TestType>(stack(ts, 0), make<TestType>({1, 3}, {0.0, 1.0, 2.0}));
        check<TestType>(stack(ts, 1), make<TestType>({3, 1}, {0.0, 1.0, 2.0}));
    }

    SECTION("empty input") {
        REQUIRE_THROWS_AS(stack(std::vector<Tensor>{}), error_t);
    }

    SECTION("different shapes") {
        std::vector<Tensor> ts{make<TestType>({3}, {0.0, 1.0, 2.0}),
                               make<TestType>({2}, {3.0, 4.0})};
        REQUIRE_THROWS_AS(stack(ts), error_t);
    }

    SECTION("different ranks") {
        std::vector<Tensor> ts{make<TestType>({2}, {0.0, 1.0}),
                               make<TestType>({2, 1}, {2.0, 3.0})};
        REQUIRE_THROWS_AS(stack(ts), error_t);
    }
}

TEST_CASE("stack: different floating-point types") {
    std::vector<Tensor> ts{make<float>({2}, {0.0, 1.0}),
                           make<double>({2}, {2.0, 3.0})};
    REQUIRE_THROWS_AS(stack(ts), std::runtime_error);
}
